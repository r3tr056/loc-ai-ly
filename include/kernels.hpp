
#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(err) cuda_check(err, __FILE__, __LINE__)

void cuda_check(cudaError_t err, const char* file, int line);

namespace Kernels {
	namespace Quantization {
		__global__ void f32_to_f16_kernel(half* output, const float* input, int size);
		__global__ void f16_to_f32_kernel(float* output, const half* input, int size);
	};

	namespace ModelKernels {
		void rmsnorm_gemm_kernel(void* output, const void* input, const float* gamma, const void* weight, float eps, int rows, int cols_in, int cols_c, ggml_type type);
        void rmsnorm_kernel(void* output, const void* input, const float* gamma, float eps, int rows, int cols, ggml_type type);
		void rope_kernel(void* x, const float* cos_cached, const float* sin_cached, int start_pos, int seq_len, int dim, ggml_type type);
		void swiglu_kernel(void* gate, void* up, int size, ggml_type type);
		void softmax_kernel(void* output, const void* input, int size, ggml_type type);
		void fused_attention_kernel(
            void* output,
            const void* query,
            const void* key_cache,
            const void* value_cache,
            int rows_q,
            int q_dim,
            int kv_dim,
            int cache_pos,
            int head_dim,
            int num_heads,
            int num_kv_heads,
            int num_groups,
            ggml_type type
        );
	}
}

#endif
