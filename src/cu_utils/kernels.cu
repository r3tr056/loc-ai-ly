#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <iostream>
#include <model_manager/gguf_loader.hpp>

void cuda_check(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in " << file << ":" << line << std::endl;
		exit(1);
	}
}

namespace Kernels {
    namespace ModelKernels {
        __global__ void f32_to_f16_kernel(half* output, const float* input, int size) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                output[i] = __float2half(input[i]);
            }
        }

        __global__ void f16_to_f32_kernel(float* output, const half* input, int size) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                output[i] = __half2float(input[i]);
            }
        }

        void rmsnorm_kernel(void* output, const void* input, const float* gamma, float eps, int rows, int cols, ggml_type type) {
            if (type == GGML_TYPE_F16) {
                Kernels::ModelKernels::rmsnorm_kernel_impl<half><<<dim3((cols + 31) / 32, (rows + 7) / 8), dim3(32, 8)>>>((half*)output, (const half*)input, gamma, eps, rows, cols);
            } else if (type == GGML_TYPE_BF16) { // BF16 dispatch
                Kernels::ModelKernels::rmsnorm_kernel_impl<__bf16><<<dim3((cols + 31) / 32, (rows + 7) / 8), dim3(32, 8)>>>((__bf16*)output, (const __bf16*)input, gamma, eps, rows, cols);
            }
            else {
                Kernels::ModelKernels::rmsnorm_kernel_impl<float><<<dim3((cols + 31) / 32, (rows + 7) / 8), dim3(32, 8)>>>((float*)output, (const float*)input, gamma, eps, rows, cols);
            }
        }

        template <typename T>
        __global__ void rmsnorm_kernel_impl(T* output, const T* input, const float* gamma, float eps, int rows, int cols) {
            int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
            if (row_idx >= rows) return; // Row-major processing

            const int threads_per_row = blockDim.x * blockDim.y;
            const int thread_idx_in_row = threadIdx.x + threadIdx.y * blockDim.x;

            __shared__ double shared_sum_sq[256];
            __shared__ float shared_scale;

            double local_sum_sq = 0.0;
            for (int col_offset = thread_idx_in_row; col_offset < cols; col_offset += threads_per_row) {
                double val = static_cast<double>(input[row_idx * cols + col_offset]);
                local_sum_sq += val * val;
            }

            shared_sum_sq[thread_idx_in_row] = local_sum_sq;
            __syncthreads();

            if (thread_idx_in_row < 256 && threads_per_row <= 256) {
                for (int i = threads_per_row / 2; i > 0; i >>= 1) {
                    if (thread_idx_in_row < i) {
                        shared_sum_sq[thread_idx_in_row] += shared_sum_sq[thread_idx_in_row + i];
                    }
                    __syncthreads();
                }
            }

            if (thread_idx_in_row == 0) {
                float scale = 1.0f / sqrtf(static_cast<float>(shared_sum_sq[0] / cols) + eps);
                shared_scale = scale;
            }
            __syncthreads();

            float scale = shared_scale;
            for (int col_offset = thread_idx_in_row; col_offset < cols; col_offset += threads_per_row) {
                output[row_idx * cols + col_offset] = static_cast<T>(static_cast<float>(input[row_idx * cols + col_offset]) * scale + gamma[col_offset]);
            }
        }

        void rmsnorm_gemm_kernel(void* output, const void* input, const float* gamma, const void* weight, float eps, int rows, int cols_in, int cols_c, ggml_type type) {
            if (type == GGML_TYPE_F16) {
                Kernels::ModelKernels::rmsnorm_gemm_kernel_impl<half><<<dim3(32, (rows + 7) / 8), dim3(32, 8)>>>((half*)output, (const half*)input, gamma, (const half*)weight, eps, rows, cols_in, cols_c);
            } else if (type == GGML_TYPE_BF16) {
                Kernels::ModelKernels::rmsnorm_gemm_kernel_impl<__bf16><<<dim3(32, (rows + 7) / 8), dim3(32, 8)>>>((__bf16*)output, (const __bf16*)input, gamma, (__bf16*)weight, eps, rows, cols_in, cols_c);
            } else { // Default to F32
                Kernels::ModelKernels::rmsnorm_gemm_kernel_impl<float><<<dim3(32, (rows + 7) / 8), dim3(32, 8)>>>((float*)output, (const float*)input, gamma, (const float*)weight, eps, rows, cols_in, cols_c);
            }
        }

        template <typename T>
        __global__ void rmsnorm_gemm_kernel_impl(T* output, const T* input, const float* gamma, const T* weight, float eps, int rows, int cols_in, int cols_c) {
            int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
            if (row_idx >= rows) return; // row-mahor processing

            const int threads_per_row = blockDim.x * blockDim.y;
            const int thread_idx_in_row = threadIdx.x + threadIdx.y * blockDim.x;

            __shared__ double shared_sum_sq[256];
            __shared__ float shared_scale;

            // calculate sum of squares within each thread block
            double local_sum_sq = 0.0;
            for (int col_offset = thread_idx_in_row; col_offset < cols; col_offset += threads_per_row) {
                double val = static_cast<double>(input[row_idx * cols + col_offset]);
                local_sum_sq += val * val;
            }

            // reduce sum of squares within the block using shared memory
            shared_sum_sq[thread_idx_in_row] = local_sum_sq;
            __syncthreads();

            if (thread_idx_in_row < 256 && threads_per_row <= 256) {
                for (int i = threads_per_row / 2; i > 0; i >>= 1) {
                    if (thread_idx_in_row < i) {
                        shared_sum_sq[thread_idx_in_row] += shared_sum_sq[thread_idx_in_row + i];
                    }
                    __syncthreads();
                }
            }

            // Calculate scale and store it in shared memory (only thread 0 in the block does this)
            if (thread_idx_in_row == 0) {
                float scale = 1.0f / sqrt(static_cast<float>(shared_sum_sq[0] / cols) + eps);
                shared_scale = scale;
            } 
            __syncthreads();

            // Fused GEMM
            float rms_scale = shared_scale;
            for (int c_col = threadIdx.x; c_col < cols_c; c_col += blockDim.x) {
                double sum = 0.0;  // Accumulate in double precision
                for (int k = 0; k < cols_in; ++k) {
                    float normalized_input_val = static_cast<float>(input[row_idx * cols_in + k]) * rms_scale + gamma[k];  // apply RMSNorm and gamma
                    sum += static_cast<double>(normalized_input_val) * static_cast<double>(weight[k * cols_c + c_col]);  // GEMM operation
                }
                output[row_idx * cols_c + c_col] = static_cast<T>(sum);  // cast back to type T for output
            }
            
        }

        void rope_kernel(void* x, const float* cos_cached, const float* sin_cached, int start_pos, int seq_len, int dim, ggml_type type) {
            if (type == GGML_TYPE_F16) {
                Kernels::ModelKernels::rope_kernel_impl<half><<<dim3((seq_len + 255) / 256), dim3(256)>>>((half*)x, cos_cached, sin_cached, start_pos, seq_len, dim);
            } else if (type == GGML_TYPE_BF16) {
                Kernels::ModelKernels::rope_kernel_impl_bf16<__bf16><<<dim3((seq_len + 255) / 256), dim3(256)>>>((__bf16*)x, cos_cached, sin_cached, start_pos, seq_len, dim);
            } else { // Default to F32
                Kernels::ModelKernels::rope_kernel_impl<float><<<dim3((seq_len + 255) / 256), dim3(256)>>>((float*)x, cos_cached, sin_cached, start_pos, seq_len, dim);
            }
        }

        // BF16 RoPE Kernel
        template <typename T>
        __global__ void rope_kernel_impl_bf16(T* x, const float* cos_cached, const float* sin_cached, int start_pos, int seq_len, int dim) {
            int pos = blockIdx.x * blockDim.x + threadIdx.x;
            if (pos >= seq_len) return;

            T* x_ptr = &x[pos * dim];

            #pragma unroll
            for (int i = 0; i < dim / 2; ++i) {
                float x0 = __bfloat162float(x_ptr[i]);
                float x1 = __bfloat162float(x_ptr[dim / 2 + i]);
                float cos_val = cos_cached[(start_pos + pos) * (dim / 2) + i];
                float sin_val = sin_cached[(start_pos + pos) * (dim / 2) + i];

                x_ptr[i] = static_cast<T>(__float2bfloat16(x0 * cos_val - x1 * sin_val));
                x_ptr[dim / 2 + i] = static_cast<T>(__float2bfloat16(x0 * sin_val + x1 * cos_val));
            }
        }

        template <typename T>
        __global__ void rope_kernel_impl(T* x, const float* cos_cached, const float* sin_cached, int start_pos, int seq_len, int dim) {
            int pos = blockIdx.x * blockDim.x + threadIdx.x;
            if (pos >= seq_len) return;
            
            T* x_ptr = &x[pos * dim];

            #pragma unroll
            for (int i = 0; i < dim / 2; ++i) {
                float x0 = static_cast<float>(x_ptr[i]);
                float x1 = static_cast<float>(x_ptr[dim / 2 + i]);
                float cos_val = cos_cached[(start_pos + pos) * (dim / 2) + i];
                float sin_val = sin_cached[(start_pos + pos) * (dim / 2) + i];

                x_ptr[i] = static_cast<T>(x0 * cos_val - x1 * sin_val);
                x_ptr[dim / 2 + i] = static_cast<T>(x0 * sin_val + x1 * cos_val);
            }
        }

        // SwiGLU kernel dispatcher
        void swiglu_kernel(void* gate, void* up, int size, ggml_type type) {
            if (type == GGML_TYPE_F16) {
                Kernels::ModelKernels::swiglu_kernel_impl<half><<<dim3((size + 255) / 256), dim3(256)>>>((half*)gate, (half*)up, size);
            } else if (type == GGML_TYPE_BF16) {
                Kernels::ModelKernels::swiglu_kernel_impl_bf16<__bf16><<<dim3((size + 255) / 256), dim3(256)>>>((__bf16*)gate, (__bf16*)up, size);
            } else { // Default to F32
                Kernels::ModelKernels::swiglu_kernel_impl<float><<<dim3((size + 255) / 256), dim3(256)>>>((float*)gate, (float*)up, size);
            }
        }

        // F32 and F16 SwiGLU kernel
        template <typename T>
        __global__ void swiglu_kernel_impl(T* gate, T* up, int size) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= size) return;

            float gate_val = static_cast<float>(gate[i]);
            float up_val = static_cast<float>(up[i]);

            float sigmoid_val = 1.0f / (1.0f + expf(-gate_val));
            gate[i] = static_cast<T>(gate_val * sigmoid_val * up_val);
        }

        // BF16 SwiGLU Kernel
        template <typename T>
        __global__ void swiglu_kernel_impl_bf16(T* gate, T* up, int size) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                float gate_val = __bfloat162float(gate[i]);
                float up_val = __bfloat162float(up[i]);
                gate[i] = static_cast<T>(__float2bfloat16(gate_val * (1.0f / (1.0f + expf(-gate_val))) * up_val));
            }
        }

        // SwiGLU gemm fused kernel implementation
        template <typename T>
        __global__ void swiglu_kernel_impl(T* output, const T* input, const T* gate_weight, const T* up_weight, const T* down_weight, int rows, int hidden_size, int intermediate_size) {

            int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
            int hidden_col_idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (row_idx >= rows || hidden_col_idx >= hidden_size) return;
            double down_proj_sum = 0.0;  // accumulate down_proj GEMM in double precision

            for (int inter_col = 0; inter_col < intermediate_size; ++inter_col) {
                float gate_val = 0.0f;
                float up_val = 0.0f;

                // --- gate projection (gemm fused)
                gate_val = static_cast<float>(input[row_idx * hidden_size + hidden_col_idx]) * static_cast<float>(gate_weight[hidden_col_idx * intermediate_size + inter_col]);
            }
        }

        void softmax_kernel(void* output, const void* input, int size, ggml_type type) {
            if (type == GGML_TYPE_F16) {
                Kernels::ModelKernels::softmax_kernel_impl<half><<<dim3((size + 255) / 256), dim3(256), 2 * 256 * sizeof(float)>>>((half*)output, (const half*)input, size);
            } else if (type == GGML_TYPE_BF16) {
                Kernels::ModelKernels::softmax_kernel_impl_bf16<__bf16><<<dim3((size + 255) / 256), dim3(256), 2 * 256 * sizeof(float)>>>((__bf16*)output, (const __bf16*)input, size);
            } else { // Default to F32
                Kernels::ModelKernels::softmax_kernel_impl<float><<<dim3((size + 255) / 256), dim3(256), 2 * 256 * sizeof(float)>>>((float*)output, (const float*)input, size);
            }
        }

        template <typename T>
        __global__ void softmax_kernel_impl(T* output, const T* input, int size) {
            extern __shared__ float shared_data[];
            float* shared_max = shared_data;
            float* shared_sum = shared_data + blockDim.x;

            int tid = threadIdx.x;
            int i = blockIdx.x * blockDim.x + tid;

            float val = (i < size) ? static_cast<float>(input[i]) : -3.402823466e+38F; // -INF for padding
            shared_max[tid] = val;

            for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
                __syncthreads();
                if (tid < stride) {
                    shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
                }
            }
            __syncthreads();
            float block_max = shared_max[0];

            if (tid == 0) {
                shared_sum[0] = 0.0f;
            }
            __syncthreads();

            if (i < size) {
                output[i] = static_cast<T>(expf(static_cast<float>(input[i]) - block_max));
            } else {
                output[i] = static_cast<T>(0.0f); // Padding output
            }
            atomicAdd(shared_sum, (i < size) ? static_cast<float>(output[i]) : 0.0f);
            __syncthreads();

            if (i < size) {
                output[i] = static_cast<T>(static_cast<float>(output[i]) / shared_sum[0]);
            }
        }

        // BF16 Softmax Kernel
        template <typename T> // Keep template for potential mixed-precision within BF16 kernels later
        __global__ void softmax_kernel_impl_bf16(T* output, const T* input, int size) {
            // ... (BF16 Softmax kernel implementation - similar to F16, but using __bf16 and BF16 intrinsics) ...
            extern __shared__ float shared_data[]; // Shared memory remains float for numerical stability
            float* shared_max = shared_data;
            float* shared_sum = shared_data + blockDim.x;

            int tid = threadIdx.x;
            int i = blockIdx.x * blockDim.x + tid;

            float val = (i < size) ? __bfloat162float(input[i]) : -3.402823466e+38F; // -INF for padding
            shared_max[tid] = val;

            for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
                __syncthreads();
                if (tid < stride) {
                    shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
                }
            }
            __syncthreads();
            float block_max = shared_max[0];

            if (tid == 0) {
                shared_sum[0] = 0.0f;
            }
            __syncthreads();

            if (i < size) {
                output[i] = static_cast<T>(__float2bfloat16(expf(__bfloat162float(input[i]) - block_max)));
            } else {
                output[i] = static_cast<T>(__float2bfloat16(0.0f)); // Padding output
            }
            atomicAdd(shared_sum, (i < size) ? __bfloat162float(output[i]) : 0.0f);
            __syncthreads();

            if (i < size) {
                output[i] = static_cast<T>(__float2bfloat16(__bfloat162float(output[i]) / shared_sum[0]));
            }
        }

        template <typename T>
        __global__ void fused_attention_kernel_impl(T* output, const T* query, const T* key_cache, const T* value_cache, int rows_q, int q_dim, int kv_dim, int cache_pos, int head_dim, int num_heads, int num_kv_heads, int num_groups) {

            int row_q_idx = blockIdx.y * blockDim.y + threadIdx.y;
            int head_idx = blockIdx.x;
            if (row_q_idx >= rows_q || head_idx >= num_heads) return;

            const int head_size = head_dim;
            const int kv_h = head_idx / num_groups;

            __shared__ float shared_data[2 * 256];
            float* shared_max = shared_data;
            float* shared_sum = shared_data + blockDim.x;
            float block_max = -3.402823466e+38F; // INF
            float block_sum = 0.0f;

            for (int k_pos = threadIdx.x; k_pos < cache_pos; k_pos += blockDim.x) {
                float score = 0.0f;
                for (int d = 0; d < head_size; ++d) {
                    score += static_cast<float>(query[row_q_idx * q_dim + head_idx * head_size + d]) * static_cast<float>(key_cache[k_pos * kv_dim + kv_h * head_size + d]);
                }
                shared_data[threadIdx.x] = score;

                // softmax - max reduction within block
                block_max = max(block_max, score);
            }

            // block-level max reduction using shared memory
            shared_max[threadIdx.x] = block_max;
            __syncthreads();

            if (threadIdx.x < 256) {
                for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (threadIdx.x < stride) {
                        shared_max[threadIdx.x] = max(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
                    }
                }
            }

            __syncthreads();
            block_max = shared_max[0];  // max score for the block

            // Softmax - Sum reduction and value weighting (fused)
            for (int k_pos = threadIdx.x; k_pos < cache_pos; k_pos += blockDim.x) {
                float score = static_cast<float>(query[row_q_idx * q_dim + head_idx * head_size + threadIdx.x]) * static_cast<float>(key_cache[k_pos * kv_dim + kv_h * head_size + threadIdx.x]);

                float softmax_score = expf(score - block_max);
                shared_data[threadIdx.x] = softmax_score;

                block_sum += softmax_score;
            }

            // block leve sum reduction using shared memory
            shared_sum[threadIdx.x] = block_sum;
            __syncthreads();

            if (threadIdx.x < 256) {
                for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (threadIdx.x < stride) {
                        shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
                    }
                }
            }
            __syncthreads();
            block_sum = shared_sum[0];

            float weighted_sum = 0.0f;
            for (int k_pos = 0; k_pos < cache_pos; ++k_pos) {
                float score = static_cast<float>(query[row_q_idx * q_dim + head_idx * head_size + threadIdx.x]) * static_cast<float>(key_cache[k_pos * kv_dim + kv_h * head_size + threadIdx.x]); // Re-calculate score - TODO: Optimize to avoid re-calculation if possible
                float softmax_score = expf(score - block_max) / block_sum; // Softmax score (finalized with block_sum)
                weighted_sum += softmax_score * static_cast<float>(value_cache[k_pos * kv_dim + kv_h * head_size + threadIdx.x]); // Weighted sum
            }

            output[row_q_idx * q_dim + head_idx * head_size + threadIdx.x] = static_cast<T>(weighted_sum);
        }

        void fused_attention_kernel(void* output, const void* query, const void* key_cache, const void* value_cache, int rows_q, int q_dim, int kv_dim, int cache_pos, int head_dim, int num_heads, int num_kv_heads, int num_groups, ggml_type type) {
            
            dim3 blockDimAttn(256);
            dim3 gridDimAttn(num_heads, (rows_q + 7) / 8);
        
            if (type == GGML_TYPE_F16) {
                Kernels::ModelKernels::fused_attention_kernel_impl<half><<<gridDimAttn, blockDimAttn, 2 * 256 * sizeof(float)>>>((half*)output, (const half*)query, (const half*)key_cache, (const half*)value_cache,rows_q, q_dim, kv_dim, cache_pos, head_dim, num_heads, num_kv_heads, num_groups);
            } else if (type == GGML_TYPE_BF16) {
                Kernels::ModelKernels::fused_attention_kernel_impl<__bf16><<<gridDimAttn, blockDimAttn, 2 * 256 * sizeof(float)>>>((__bf16*)output, (const __bf16*)query, (const __bf16*)key_cache, (const __bf16*)value_cache, rows_q, q_dim, kv_dim, cache_pos, head_dim, num_heads, num_kv_heads, num_groups);
            }
            else { // Default to F32
                Kernels::ModelKernels::fused_attention_kernel_impl<float><<<gridDimAttn, blockDimAttn, 2 * 256 * sizeof(float)>>>((float*)output, (const float*)query, (const float*)key_cache, (const float*)value_cache, rows_q, q_dim, kv_dim, cache_pos, head_dim, num_heads, num_kv_heads, num_groups);
            }
        }
    }
}