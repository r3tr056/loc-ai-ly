#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <iostream>

namespace Kernels {
    namespace Quantization {
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
    }
}