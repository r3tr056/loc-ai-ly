#include <vector>
#include <model_manager/gguf_loader.hpp>
#include <kernels.hpp>


// Function to quantize a float tensor to INT8 (simple symmetric quantization)
std::vector<int8_t> quantize_f32_to_int8(const std::vector<float>& float_data) {
    if (float_data.empty()) return {};

    float max_val = 0.0f;
    for (float val : float_data) {
        max_val = std::max(max_val, std::abs(val));
    }
    float scale = 127.0f / max_val; // Scale factor for symmetric quantization

    std::vector<int8_t> int8_data(float_data.size());
    for (size_t i = 0; i < float_data.size(); ++i) {
        int8_data[i] = static_cast<int8_t>(roundf(float_data[i] * scale));
    }
    return int8_data;
}

// Function to dequantize INT8 tensor back to float
std::vector<float> dequantize_int8_to_f32(const std::vector<int8_t>& int8_data) {
	if (int8_data.empty()) return {};

	float max_abs_val = 0.0f;  // Need to determine scale to dequantize properly. For simplicity, we assume a fixed scale for now. In practice, scale should be stored per tensor or block.
    // In a real implementation, you would need to store and retrieve the scale factor used during quantization.
    // For this simple example, we'll use a placeholder scale.
	float placeholder_scale = 1.0f;
	for (int8_t val : int8_data) {
		max_abs_val = std::max(max_abs_val, static_cast<float>(std::abs(val)));
	}
	float scale = max_abs_val > 0 ? max_abs_val / 127.0f : 1.0f;  // Estimate scale back - This is a simplification.

	std::vector<float> float_data(int8_data.size());
	for (size_t i = 0; i < int8_data.size(); ++i) {
		float_data[i] = static_cast<float>(int8_data[i]) / scale;
	}
	return float_data;
}


std::vector<float> half_to_float_host(const std::vector<half>& half_data) {
    std::vector<float> float_data(half_data.size());
    for (size_t i = 0; i < half_data.size(); ++i) {
        float_data[i] = __half2float(half_data[i]);
    }
    return float_data;
}

std::vector<half> float_to_half_host(const std::vector<float>& float_data) {
    std::vector<half> half_data(float_data.size());
    for (size_t i = 0; i < float_data.size(); ++i) {
        half_data[i] = __float2half(float_data[i]);
    }
    return half_data;
}

void float_to_half_gpu(half* output, const float* input, size_t size) {
    dim3 blockDimConvert(256);
    dim3 gridDimConvert((size + blockDimConvert.x - 1) / blockDimConvert.x);
    Kernels::Quantization::f32_to_f16_kernel<<<gridDimConvert, blockDimConvert>>>(output, input, size);
    CHECK_CUDA(cudaDeviceSynchronize());
}

// GPU-accelerated half to float conversion
void half_to_float_gpu(float* output, const half* input, size_t size) {
    dim3 blockDimConvert(256);
    dim3 gridDimConvert((size + blockDimConvert.x - 1) / blockDimConvert.x);
    Kernels::Quantization::f16_to_f32_kernel<<<gridDimConvert, blockDimConvert>>>(output, input, size);
    CHECK_CUDA(cudaDeviceSynchronize());
}