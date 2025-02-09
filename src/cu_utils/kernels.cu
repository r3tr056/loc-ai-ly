#include <cuda_runtime.h>
#include <cmath>

namespace Kernels {
    __global__ void rmsnorm_kernel(float* output, const float* input, const float* gamma, float eps, int rows, int cols) {
        int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
        int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (row_idx < rows && col_idx < cols) {
            int base_idx = row_idx * cols;
            if (col_idx == 0) { // First thread in each row calculates row sum_sq
                float sum_sq = 0.0f;
                for (int j = 0; j < cols; ++j) {
                    sum_sq += input[base_idx + j] * input[base_idx + j];
                }
                float scale = 1.0f / sqrtf(sum_sq / cols + eps);
                output[base_idx + cols] = scale; // Store scale temporarily after the row
            }
            __syncthreads(); // Ensure scale is calculated for the row before proceeding

            float scale = output[base_idx + cols]; // Retrieve the pre-calculated scale
            output[base_idx + col_idx] = (input[base_idx + col_idx] * scale + gamma[col_idx]);
        }
    }


    __global__ void rope_kernel(float* x, const float* cos_cached, const float* sin_cached, int start_pos, int seq_len, int dim) {
        int pos = blockIdx.x * blockDim.x + threadIdx.x;
        if (pos < seq_len) {
            float* x_ptr = &x[pos * dim];
            for (int i = 0; i < dim / 2; ++i) {
                float x0 = x_ptr[i];
                float x1 = x_ptr[dim / 2 + i];
                float cos_val = cos_cached[(start_pos + pos) * (dim / 2) + i];
                float sin_val = sin_cached[(start_pos + pos) * (dim / 2) + i];

                x_ptr[i] = x0 * cos_val - x1 * sin_val;
                x_ptr[dim / 2 + i] = x0 * sin_val + x1 * cos_val;
            }
        }
    }

    __global__ void swiglu_kernel(float* gate, float* up, int size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            gate[i] = gate[i] * (1.0f / (1.0f + expf(-gate[i]))) * up[i];
        }
    }

    __global__ void softmax_kernel(float* output, const float* input, int size) {
        extern __shared__ float shared_data[];
        float* shared_max = shared_data;
        float* shared_sum = shared_data + blockDim.x;

        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x + tid;

        float val = (i < size) ? input[i] : -3.402823466e+38F; // -INF for padding
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
            output[i] = expf(input[i] - block_max);
        } else {
            output[i] = 0.0f; // Padding output
        }
        atomicAdd(shared_sum, (i < size) ? output[i] : 0.0f);
        __syncthreads();

        if (i < size) {
            output[i] /= shared_sum[0];
        }
    }
}