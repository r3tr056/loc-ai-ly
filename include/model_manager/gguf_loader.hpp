#ifndef GGUF_LOADER_H
#define GGUF_LOADER_H

#include <cmath>
#include <vector>
#include <cstdint>
#include <fstream>
#include <cstring>
#include <iostream>
#include <memory>
#include <cuda_fp16.h>
#include <vector_types.h>

// GGUF enums and structs (from provided spec)
enum ggml_type : uint32_t {
	GGML_TYPE_F32 = 0,
	GGML_TYPE_F16 = 1,
	GGML_TYPE_Q4_0 = 2,
	GGML_TYPE_Q4_1 = 3,
	GGML_TYPE_Q5_0 = 6,
	GGML_TYPE_Q5_1 = 7,
	GGML_TYPE_Q8_0 = 8,
	GGML_TYPE_Q8_1 = 9,
	GGML_TYPE_Q2_K = 10,
	GGML_TYPE_Q3_K = 11,
	GGML_TYPE_Q4_K = 12,
	GGML_TYPE_Q5_K = 13,
	GGML_TYPE_Q6_K = 14,
	GGML_TYPE_Q8_K = 15,
	GGML_TYPE_IQ2_XXS = 16,
	GGML_TYPE_IQ2_XS = 17,
	GGML_TYPE_IQ3_XXS = 18,
	GGML_TYPE_IQ1_S = 19,
	GGML_TYPE_IQ4_NL = 20,
	GGML_TYPE_IQ3_S = 21,
	GGML_TYPE_IQ2_S = 22,
	GGML_TYPE_IQ4_XS = 23,
	GGML_TYPE_I8 = 24,
	GGML_TYPE_I16 = 25,
	GGML_TYPE_I32 = 26,
	GGML_TYPE_I64 = 27,
	GGML_TYPE_F64 = 28,
	GGML_TYPE_IQ1_M = 29,
	GGML_TYPE_BF16 = 30,
	GGML_TYPE_COUNT,
};

enum gguf_metadata_value_type : uint32_t {
	GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
	GGUF_METADATA_VALUE_TYPE_INT8 = 1,
	GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
	GGUF_METADATA_VALUE_TYPE_INT16 = 3,
	GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
	GGUF_METADATA_VALUE_TYPE_INT32 = 5,
	GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
	GGUF_METADATA_VALUE_TYPE_BOOL = 7,
	GGUF_METADATA_VALUE_TYPE_STRING = 8,
	GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
	GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
	GGUF_METADATA_VALUE_TYPE_INT64 = 11,
	GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

struct gguf_string_t {
	uint64_t len;
	char *string; // Dynamically allocated string
};

union gguf_metadata_value_t {
	uint8_t uint8;
	int8_t int8;
	uint16_t uint16;
	int16_t int16;
	uint32_t uint32;
	int32_t int32;
	float float32;
	uint64_t uint64;
	int64_t int64;
	double float64;
	bool bool_;
	gguf_string_t string;
	struct {
		gguf_metadata_value_type type;
		uint64_t len;
		gguf_metadata_value_t *array; // Dynamically allocated array
	} array;
};

struct gguf_metadata_kv_t {
	gguf_string_t key;
	gguf_metadata_value_type value_type;
	gguf_metadata_value_t value;
};

struct gguf_header_t {
	uint32_t magic;
	uint32_t version;
	uint64_t tensor_count;
	uint64_t metadata_kv_count;
};

struct gguf_tensor_info_t {
	gguf_string_t name;
	uint32_t n_dimensions;
	uint64_t dimensions[4]; // Max 4 dimensions as per spec
	ggml_type type;
	uint64_t offset;
};

std::string read_gguf_string(std::ifstream& file);

std::vector<int8_t> quantize_f32_to_int8(const std::vector<float>& float_data);
std::vector<float> dequantize_int8_to_f32(const std::vector<int8_t>& int8_data);

// host side float to half conversion (for initial loading if needed)
std::vector<half> float_to_half_host(const std::vector<float>& float_data);
// host side half to float conversion
std::vector<float> half_to_float_host(const std::vector<half>& half_data);

void float_to_half_gpu(half* output, const float* input, size_t size);
void half_to_float_gpu(float* output, const half* input, size_t size);

// std::vector<__bf16> float_to_bf16_host(const std::vector<float>& float_data) {
// 	std::vector<__bf16> bf16_data(float_data.size());
// 	for (size_t i = 0; i < float_data.size(); ++i) {
// 		bf16_data[i] = __float2bfloat16(float_data[i]);
// 	}
// 	return bf16_data;
// }

// std::vector<float> bf16_to_float_host(const std::vector<__bf16>& bf16_data) {
// 	std::vector<float> float_data(bf16_data.size());
// 	for (size_t i = 0; i < bf16_data.size(); ++i) {
// 		float_data[i] = __bfloat162float(bf16_data[i]);
// 	}
// 	return float_data;
// }

void free_gguf_metadata_value(std::pair<gguf_metadata_value_t, gguf_metadata_value_type>& value);

std::pair<gguf_metadata_value_t, gguf_metadata_value_type> read_gguf_metadata_value(std::ifstream& file);

#endif  // GGUF_LOADER_H