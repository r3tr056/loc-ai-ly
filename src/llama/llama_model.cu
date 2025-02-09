#include <vector>
#include <cmath>
#include <string>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <cstring>
#include <map>
#include <stdexcept>
#include <sstream>
#include <memory>
#include <algorithm>
#include <cblas.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <iomanip>


#define CHECK_CUDA(err) cuda_check(err, __FILE__, __LINE__)

void cuda_check(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in " << file << ":" << line << std::endl;
		exit(1);
	}
}

namespace Kernels {
	__global__ void rmsnorm_kernel(float* output, const float* input, const float* gamma, float eps, int rows, int cols);
	__global__ void rope_kernel(float* x, const float* cos_cached, const float* sin_cached, int start_pos, int seq_len, int dim);
	__global__ void swiglu_kernel(float* gate, float* up, int size);
	__global__ void softmax_kernel(float* output, const float* input, int size);
}

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

// Llama3 8B config
struct LlamaConfig {
	int vocab_size = 128256;
	int hidden_size = 4096;
	int intermediate_size = 14336;
	int num_hidden_layers = 32;
	int num_attention_heads = 32;
	int num_key_value_heads = 8;
	float rms_norm_eps = 1e-5;
	int max_position_embedding = 8192;

	int head_dim() const { return hidden_size / num_attention_heads; }

	static LlamaConfig from_metadata(const std::map<std::string, std::pair<gguf_metadata_value_t, gguf_metadata_value_type>>& metadata);
};

LlamaConfig LlamaConfig::from_metadata(const std::map<std::string, std::pair<gguf_metadata_value_t, gguf_metadata_value_type>>& metadata) {
	LlamaConfig config;
	auto get_int = [&](const std::string& key, int& val) {
		if (metadata.count(key)) {
			gguf_metadata_value_type type = metadata.at(key).second;
			if (type == GGUF_METADATA_VALUE_TYPE_UINT32)
				val = metadata.at(key).first.uint32;
			else if (type == GGUF_METADATA_VALUE_TYPE_INT32)
				val = metadata.at(key).first.int32;
			else if (type == GGUF_METADATA_VALUE_TYPE_UINT64)
				val = static_cast<int>(metadata.at(key).first.uint64); // Explicit cast, handle potential overflow
			else if (type == GGUF_METADATA_VALUE_TYPE_INT64)
				val = static_cast<int>(metadata.at(key).first.int64); // Explicit cast, handle potential overflow
			else {
				std::cerr << "Warning: Metadata key '" << key << "' has unexpected type: " << type << ", expected integer." << std::endl;
			}
		}
	};

	auto get_float = [&](const std::string& key, float& val) {
		if (metadata.count(key)) {
			gguf_metadata_value_type type = metadata.at(key).second;
			if (type == GGUF_METADATA_VALUE_TYPE_FLOAT32)
				val = metadata.at(key).first.float32;
			else if (type == GGUF_METADATA_VALUE_TYPE_FLOAT64)
				val = static_cast<float>(metadata.at(key).first.float64); // Explicit cast from double to float
			else {
				std::cerr << "Warning: Metadata key '" << key << "' has unexpected type: " << type << ", expected float." << std::endl;
			}
		}
	};

	get_int("tokenizer.ggml.vocabulary_size", config.vocab_size);
    get_int("llama.embedding_length", config.hidden_size);
    get_int("llama.feed_forward_length", config.intermediate_size);
    get_int("llama.block_count", config.num_hidden_layers);
    get_int("llama.attention.head_count", config.num_attention_heads);
    get_int("llama.attention.head_count_kv", config.num_key_value_heads);
    get_float("llama.attention.layer_norm_rms_epsilon", config.rms_norm_eps);
    get_int("llama.context_length", config.max_position_embedding);


    if (config.hidden_size <= 0 || config.num_hidden_layers <= 0 || config.num_attention_heads <= 0) {
        throw std::runtime_error("Error: Invalid model configuration loaded from metadata. Check GGUF file.");
    }
    return config;

};

// GPU memory managerment
struct GPUData {
	float* data;
	size_t size;

	GPUData(size_t elements) : size(elements) { cudaMalloc(&data, elements * sizeof(float)); }
	~GPUData() { cudaFree(data); }
};

// GPU tensor
struct Tensor {
	std::vector<int> shape;
	GPUData gpu_data;

	Tensor(const std::vector<int>& dims) : shape(dims), gpu_data(num_elements()) {}
	Tensor(const std::vector<int>& dims, float* external_data) : shape(dims), gpu_data(0) {
		gpu_data.data = external_data;
	}

	size_t num_elements() const {
		size_t count = 1;
		for (int dim : shape) count *= dim;
		return count;
	}
};

// model weights
struct ModelWeights {
	std::unordered_map<std::string, Tensor> tensors;
	LlamaConfig config;
};


// Matrix Operations (BLAS)
struct Matrix {
	GPUData gpu_data;
	int rows, cols;
	cublasHandle_t cublas_handle;
	bool owns_data;

	Matrix(int r, int c, cublasHandle_t handle) : rows(r), cols(c), cublas_handle(handle), gpu_data(r * c), owns_data(true) {
		CHECK_CUDA(cudaMemset(gpu_data.data, 0, r * c * sizeof(float)));
	}

	Matrix(const std::vector<int>& dims, float* external_data, cublasHandle_t handle, bool _owns_data = false) : rows(dims[0]), cols(dims[1]), cublas_handle(handle), gpu_data(0), owns_data(_owns_data) {
		if (dims.size() != 2) {
			throw std::runtime_error("Matrix constructor with external data expected 2 dims.");
		}
		gpu_data.data = external_data;
	}

	~Matrix() {
		if (!owns_data) {
			gpu_data.data = nullptr;
			gpu_data.size = 0;
		}
	}

	float* data() {return gpu_data.data;}

	void gemm(const Matrix& a, const Matrix& b, bool transpose_a = false, bool transpose_b = false) {
		const float alpha = 1.0f;
		const float beta = 0.0f;
		cublasOperation_t transA = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t transB = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

		// TODO : Creat macros for checking status and errors of cuda operations and handle the errors
		cublasSgemm_v2(
			cublas_handle,
			transB, transA,
			cols, rows, a.cols, // m, n, k  (output rows, output cols, inner dim) - adjusted for row-major logic and swapped A/B
			&alpha,
			b.gpu_data.data, b.cols, // B (transposed if needed) - adjusted for row-major logic and swapped A/B
			a.gpu_data.data, a.cols, // A (transposed if needed) - adjusted for row-major logic and swapped A/B
			&beta,
			gpu_data.data, cols
		);
	}
};

// RMS Normalization
class RMSNorm {
public:
	Matrix gamma;
	float eps;
	cublasHandle_t cublas_handle;

public:
	RMSNorm(int dim, float eps, cublasHandle_t handle) : gamma(1, dim, handle), eps(eps), cublas_handle(handle) {
		// initialize gamma
		float* h_gamma = new float[dim];
		std::fill(h_gamma, h_gamma + dim, 1.0f);
		CHECK_CUDA(cudaMemcpy(gamma.data(), h_gamma, dim * sizeof(float), cudaMemcpyHostToDevice));
		delete[] h_gamma;
	}

	__host__ void forward(const Matrix& input, Matrix& output) {
		if (output.rows != input.rows || output.cols != input.cols) {
			output = Matrix(input.rows, input.cols, cublas_handle);
		}

		dim3 blockDim(32, 32); // FLAG: Tunable paramater
		dim3 gridDim(
			(input.cols + blockDim.x - 1) / blockDim.x,
			(input.rows + blockDim.y - 1) / blockDim.y
		);

		Kernels::rmsnorm_kernel<<<gridDim, blockDim>>>(
			output.data(),
			input.gpu_data.data,
			gamma.data(), 
			eps, 
			input.rows, 
			input.cols
		);

		// cudaError_t err = cudaGetLastError();
		// if (err != cudaSuccess) {
		//     throw std::runtime_error("CUDA kernel launch error: " + std::string(cudaGetErrorString(err)));
		// }

		CHECK_CUDA(cudaDeviceSynchronize());
	}
};

// Rotary Positional Embedding
class RotaryEmbedding {
public:
	std::vector<float> inv_freq;
	Tensor sin_cached;
	Tensor cos_cached;
	cublasHandle_t cublas_handle;

public:
	RotaryEmbedding(int dim, int max_seq_len, cublasHandle_t handle)
		: sin_cached({max_seq_len, dim / 2}),
		cos_cached({max_seq_len, dim / 2}),
		cublas_handle(handle)
	{
		inv_freq.resize(dim / 2);
		for (int i = 0; i < dim / 2; ++i) {
			inv_freq[i] = 1.0f / pow(10000.0f, 2.0f * i / dim);
		}

		std::vector<float> cpu_sin_cached(max_seq_len * (dim / 2));
		std::vector<float> cpu_cos_cached(max_seq_len * (dim / 2));

		for (int pos = 0; pos < max_seq_len; ++pos) {
			for (int i = 0; pos < dim / 2; ++i) {
				float freq = pos * inv_freq[i];
				cpu_cos_cached[pos * (dim / 2) + i] = cos(freq);
				cpu_sin_cached[pos * (dim / 2) + i] = sin(freq);
			}
		}

		CHECK_CUDA(cudaMemcpy(cos_cached.gpu_data.data, cpu_cos_cached.data(), cpu_cos_cached.size() * sizeof(float), cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(sin_cached.gpu_data.data, cpu_sin_cached.data(), cpu_sin_cached.size() * sizeof(float), cudaMemcpyHostToDevice));
	}

	void apply(Matrix& x, int start_pos) {
		dim3 blockDim(256);
		dim3 gridDim((x.rows + blockDim.x - 1) / blockDim.x);

		Kernels::rope_kernel<<<gridDim, blockDim>>>(x.data(), cos_cached.gpu_data.data, sin_cached.gpu_data.data, start_pos, x.rows, x.cols);
		cudaError_t err = cudaGetLastError();

		if (err != cudaSuccess) {
			throw std::runtime_error("CUDA kernel launch error: " + 
				std::string(cudaGetErrorString(err)));
		}
		CHECK_CUDA(cudaDeviceSynchronize());
	}
};

// Attention Layer with KV Cache
class Attention {
public:
	LlamaConfig config;
	Matrix q_proj, k_proj, v_proj, o_proj;
	RotaryEmbedding rope;

	// KV Cache
	Matrix key_cache, value_cache;
	int cache_pos = 0;
	cublasHandle_t cublas_handle;

public:
	Attention(const LlamaConfig& config, cublasHandle_t handle)
		: config(config),
		rope(config.head_dim(), config.max_position_embedding, handle),
		key_cache({config.max_position_embedding, config.num_key_value_heads * config.head_dim()}, nullptr, handle, true),
		value_cache({config.max_position_embedding, config.num_key_value_heads * config.head_dim()}, nullptr, handle, true),
		q_proj({config.hidden_size, config.num_attention_heads * config.head_dim()}, nullptr, handle, true),
		k_proj({config.hidden_size, config.num_key_value_heads * config.head_dim()}, nullptr, handle, true),
		v_proj({config.hidden_size, config.num_key_value_heads * config.head_dim()}, nullptr, handle, true),
		o_proj({config.num_attention_heads * config.head_dim(), config.hidden_size}, nullptr, handle, true),
		cublas_handle(handle)
	{
		q_proj = Matrix(config.hidden_size, config.num_attention_heads * config.head_dim(), handle);
		k_proj = Matrix(config.hidden_size, config.num_key_value_heads * config.head_dim(), handle);
		v_proj = Matrix(config.hidden_size, config.num_key_value_heads * config.head_dim(), handle);
		o_proj = Matrix(config.num_attention_heads * config.head_dim(), config.hidden_size, handle);
		key_cache = Matrix(config.max_position_embedding, config.num_key_value_heads * config.head_dim(), handle);
		value_cache = Matrix(config.max_position_embedding, config.num_key_value_heads * config.head_dim(), handle);
	}

	
	void forward(const Matrix& input, Matrix& output, int start_pos) {
		const int q_dim = config.num_attention_heads * config.head_dim();
		const int kv_dim = config.num_key_value_heads * config.head_dim();

		Matrix q(input.rows, q_dim, cublas_handle);
		q.gemm(input, q_proj);

		Matrix k(input.rows, kv_dim, cublas_handle);
		k.gemm(input, k_proj);

		Matrix v(input.rows, kv_dim, cublas_handle);
		v.gemm(input, v_proj);

		// apply RoPE
		rope.apply(q, start_pos);
		rope.apply(k, start_pos);

		// update kv cache
		size_t copy_size = input.rows * kv_dim * sizeof(float);
		CHECK_CUDA(cudaMemcpyAsync(&key_cache.data()[cache_pos * kv_dim], k.data(), copy_size, cudaMemcpyDeviceToDevice));
		CHECK_CUDA(cudaMemcpyAsync(&value_cache.data()[cache_pos * kv_dim], v.data(), copy_size, cudaMemcpyDeviceToDevice));
		cache_pos += input.rows; // Increment cache_pos by the number of tokens processed
		CHECK_CUDA(cudaStreamSynchronize(0)); // finish updates

		// Multi head attention
		const int num_groups = config.num_attention_heads / config.num_key_value_heads;
		Matrix attn_output(input.rows, q_dim, cublas_handle);

		for (int h = 0; h < config.num_attention_heads; ++h) {
			const int kv_h = h / num_groups;
			const int head_size = config.head_dim();

			// compute attention scores
			Matrix scores({1, cache_pos}, nullptr, cublas_handle, true);
			CHECK_CUDA(cudaMemset(scores.data(), 0, cache_pos * sizeof(float)));

			const float alpha_sgemv = 1.0f / sqrt(head_size);
			const float beta_sgemv = 0.0f;
			cublasSgemv_v2(
				cublas_handle,
				CUBLAS_OP_N,
				cache_pos, head_size,
				&alpha_sgemv,
				&key_cache.data()[kv_h * kv_dim], kv_dim,
				&q.data()[h * head_size], 1,
				&beta_sgemv,
				scores.data(), 1
			);

			// softmax (using the CUDA kernel)
			dim3 blockDimSoftmax(256);
			dim3 gridDimSoftmax((cache_pos + blockDimSoftmax.x - 1) / blockDimSoftmax.x);
			Kernels::softmax_kernel<<<gridDimSoftmax, blockDimSoftmax, 2 * blockDimSoftmax.x * sizeof(float)>>>(
				scores.data(), scores.data(), cache_pos
			);
			
			// error checking
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) {
				throw std::runtime_error("CUDA kernel launch error: " + 
					std::string(cudaGetErrorString(err)));
			}
			CHECK_CUDA(cudaDeviceSynchronize());

			// weighted sum of values
			float alpha_sgemv2 = 1.0f;
			float beta_sgemv2 = 0.0f;

			cublasSgemv_v2(
				cublas_handle,
				CUBLAS_OP_N,
				head_size, cache_pos,
				&alpha_sgemv2,
				&value_cache.data()[kv_h * kv_dim], kv_dim,
				scores.data(), 1,
				&beta_sgemv2,
				&attn_output.data()[h * head_size], 1
			);
		}

		// output projection
		output.gemm(attn_output, o_proj);
	}
};


// feed forward network (SwiGLU)
class FeedForward {
public:
	Matrix gate_proj, up_proj, down_proj;
	cublasHandle_t cublas_handle;

public:
	FeedForward(const LlamaConfig& config, cublasHandle_t handle)
		: gate_proj({config.hidden_size, config.intermediate_size}, nullptr, handle, true),
		up_proj({config.hidden_size, config.intermediate_size}, nullptr, handle, true),
		down_proj({config.intermediate_size, config.hidden_size}, nullptr, handle, true),
		cublas_handle(handle)
	{
		gate_proj = Matrix(config.hidden_size, config.intermediate_size, handle);
		up_proj = Matrix(config.hidden_size, config.intermediate_size, handle);
		down_proj = Matrix(config.intermediate_size, config.hidden_size, handle);
	}

	void forward(const Matrix& input, Matrix& output) {
		Matrix gate(input.rows, gate_proj.cols, cublas_handle);
		gate.gemm(input, gate_proj);

		Matrix up(input.rows, up_proj.cols, cublas_handle);
		up.gemm(input, up_proj);

		// SwiGLU activation (CUDA kernel)
		dim3 blockDimSwiglu(256); // FLAG: Tunable parameter
		dim3 gridDimSwiglu((gate.rows * gate.cols + blockDimSwiglu.x - 1) / blockDimSwiglu.x);
		Kernels::swiglu_kernel<<<gridDimSwiglu, blockDimSwiglu>>>(gate.data(), up.data(), gate.rows * gate.cols);

		// Error checking
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			throw std::runtime_error("CUDA kernel launch error: " + 
				std::string(cudaGetErrorString(err)));
		}
		CHECK_CUDA(cudaDeviceSynchronize());

		output.gemm(gate, down_proj);
	}
};

// Transformer block
class TransformerBlock {
public:
	Attention attn;
	FeedForward ff;
	RMSNorm input_norm, post_attn_norm;
	cublasHandle_t cublas_handle;

public:
	TransformerBlock(const LlamaConfig& config, cublasHandle_t handle)
	: attn(config, handle), ff(config, handle),
	input_norm(config.hidden_size, config.rms_norm_eps, handle),
	post_attn_norm(config.hidden_size, config.rms_norm_eps, handle),
	cublas_handle(handle) {}

	void forward(const Matrix& input, Matrix& output, int pos) {
		Matrix norm_input(input.rows, input.cols, cublas_handle);
		input_norm.forward(input, norm_input);

		Matrix attn_output(input.rows, input.cols, cublas_handle);
		attn.forward(norm_input, attn_output, pos);

		// Residual connection (element wise addition, can be CUDA kernelized)
		for (int i = 0; i < input.rows * input.cols; ++i) {
			attn_output.data()[i] += input.gpu_data.data[i];
		}

		Matrix norm_attn(attn_output.rows, attn_output.cols, cublas_handle);
		post_attn_norm.forward(attn_output, norm_attn);

		Matrix ff_output(norm_attn.rows, norm_attn.cols, cublas_handle);
		ff.forward(norm_attn, ff_output);

		// final residual connection
		for (int i = 0; i < ff_output.rows * ff_output.cols; ++i)
			output.data()[i] = attn_output.data()[i] + ff_output.data()[i];
	}
};

class LlamaModel {
public:
	std::vector<TransformerBlock> layers;
	Matrix embedding;
	RMSNorm final_norm;
	Matrix lm_head;
	int current_pos = 0;
	cublasHandle_t cublas_handle;

public:
	LlamaModel(const LlamaConfig& config, cublasHandle_t handle)
		: layers(config.num_hidden_layers, TransformerBlock(config, handle)),
		embedding({config.vocab_size, config.hidden_size}, nullptr, handle, true),
		final_norm(config.hidden_size, config.rms_norm_eps, handle),
		lm_head({config.hidden_size, config.vocab_size}, nullptr, handle, true),
		cublas_handle(handle)
	{
		embedding = Matrix(config.vocab_size, config.hidden_size, handle);
		lm_head = Matrix(config.hidden_size, config.vocab_size, handle);
	}

	void forward(const std::vector<int>& tokens, Matrix& logits) {
		Matrix token_embeddings((int)tokens.size(), embedding.cols, cublas_handle);

		for (size_t i = 0; i < tokens.size(); ++i) {
			CHECK_CUDA(cudaMemcpyAsync(&token_embeddings.data()[i * embedding.cols], &embedding.data()[tokens[i] * embedding.cols], embedding.cols * sizeof(float), cudaMemcpyDeviceToDevice));
		}
		CHECK_CUDA(cudaStreamSynchronize(0));

		Matrix current_hidden = token_embeddings;

		for (auto& layer : layers) {
			Matrix layer_output(current_hidden.rows, current_hidden.cols, cublas_handle);
			layer.forward(current_hidden, layer_output, current_pos);
			current_hidden = layer_output;
		}

		Matrix normed_hidden(current_hidden.rows, current_hidden.cols, cublas_handle);
		final_norm.forward(current_hidden, normed_hidden);
		logits.gemm(normed_hidden, lm_head);
		current_pos += tokens.size();
	}
};

std::string read_gguf_string(std::ifstream& file) {
	uint64_t len;
	file.read(reinterpret_cast<char*>(&len), sizeof(uint64_t));
	if (len > 1024 * 1024) {
		throw std::runtime_error("Error: GGUF string length too large, likely corrupted file.");
	}
	std::unique_ptr<char[]> buffer(new char[len]);
	file.read(buffer.get(), len);
	return std::string(buffer.get(), len);
}

void free_gguf_metadata_value(std::pair<gguf_metadata_value_t, gguf_metadata_value_type>& value) {
    if (value.second == GGUF_METADATA_VALUE_TYPE_STRING && value.first.string.string != nullptr) {
        delete[] value.first.string.string;
        value.first.string.string = nullptr;
    } else if (value.second == GGUF_METADATA_VALUE_TYPE_ARRAY && value.first.array.array != nullptr) {
        for (uint64_t i = 0; i < value.first.array.len; ++i) {
            std::pair<gguf_metadata_value_t, gguf_metadata_value_type> element_value = 
                { value.first.array.array[i], GGUF_METADATA_VALUE_TYPE_ARRAY };
            
            free_gguf_metadata_value(element_value);
        }
        delete[] value.first.array.array;
        value.first.array.array = nullptr;
    }
}

std::pair<gguf_metadata_value_t, gguf_metadata_value_type> read_gguf_metadata_value(std::ifstream& file) {
	gguf_metadata_value_type value_type;
	file.read(reinterpret_cast<char*>(&value_type), sizeof(gguf_metadata_value_type));
	
	gguf_metadata_value_t value;

	switch (value_type) {
		case GGUF_METADATA_VALUE_TYPE_UINT8:   file.read(reinterpret_cast<char*>(&value.uint8), sizeof(uint8_t)); break;

		case GGUF_METADATA_VALUE_TYPE_INT8:    file.read(reinterpret_cast<char*>(&value.int8), sizeof(int8_t)); break;

		case GGUF_METADATA_VALUE_TYPE_UINT16:  file.read(reinterpret_cast<char*>(&value.uint16), sizeof(uint16_t)); break;

		case GGUF_METADATA_VALUE_TYPE_INT16:   file.read(reinterpret_cast<char*>(&value.int16), sizeof(int16_t)); break;

		case GGUF_METADATA_VALUE_TYPE_UINT32:  file.read(reinterpret_cast<char*>(&value.uint32), sizeof(uint32_t)); break;

		case GGUF_METADATA_VALUE_TYPE_INT32:   file.read(reinterpret_cast<char*>(&value.int32), sizeof(int32_t)); break;

		case GGUF_METADATA_VALUE_TYPE_FLOAT32: file.read(reinterpret_cast<char*>(&value.float32), sizeof(float)); break;

		case GGUF_METADATA_VALUE_TYPE_BOOL:    file.read(reinterpret_cast<char*>(&value.bool_), sizeof(bool)); break;

		case GGUF_METADATA_VALUE_TYPE_STRING: {
			std::string str_val = read_gguf_string(file);
			value.string.len = str_val.length();
			value.string.string = new char[value.string.len + 1];   // +1 for null terminator
			std::strcpy(value.string.string, str_val.c_str());
			break;
		}

		case GGUF_METADATA_VALUE_TYPE_ARRAY: {
			file.read(reinterpret_cast<char*>(&value.array.type), sizeof(gguf_metadata_value_type));
			file.read(reinterpret_cast<char*>(&value.array.len), sizeof(uint64_t));
			if (value.array.len > 1024 * 1024) {
				throw std::runtime_error("Error: GGUF array length too large, likely corrupted file.");
			}
			value.array.array = new gguf_metadata_value_t[value.array.len];
			for (uint64_t i = 0; i < value.array.len; ++i) {
				auto [metadata_value, _] = read_gguf_metadata_value(file);
				value.array.array[i] = metadata_value;
			}
			break;
		}

		case GGUF_METADATA_VALUE_TYPE_UINT64:  file.read(reinterpret_cast<char*>(&value.uint64), sizeof(uint64_t)); break;

		case GGUF_METADATA_VALUE_TYPE_INT64:   file.read(reinterpret_cast<char*>(&value.int64), sizeof(int64_t)); break;

		case GGUF_METADATA_VALUE_TYPE_FLOAT64: file.read(reinterpret_cast<char*>(&value.float64), sizeof(double)); break;

		default:
			throw std::runtime_error("Error: Unknown GGUF metadata value type: " + std::to_string(value_type));
	}
	return {value, value_type};
}

ModelWeights load_weights(const std::string& path) {
	std::ifstream file(path, std::ios::binary);
	ModelWeights weights;
	std::map<std::string, std::pair<gguf_metadata_value_t, gguf_metadata_value_type>> metadata_map;

	// read header - GGUF Magic and Version
	gguf_header_t header;
	file.read(reinterpret_cast<char*>(&header), sizeof(gguf_header_t));

	if (header.magic != 0x47475546) {
		std::cerr << "Error: Invalid GGUF magic number: " << std::hex << header.magic << std::endl;
		exit(1);
	}

	// Check for version 1 for now
	if (header.version != 1) {
		std::cerr << "Error: Unsupported GGUF version: " << header.version << ". Expected version 1." << std::endl;
		exit(1);
	}

	// Read metadata KVs
	// TODO : Make a method read_gguf metadata that reads both key and value together
	// with peroper gguf_metadata_kv_t
	for (uint64_t i = 0; i < header.metadata_kv_count; ++i) {
		uint32_t key_len_u32;
		file.read(reinterpret_cast<char*>(&key_len_u32), sizeof(uint32_t));
		gguf_string_t key_gguf;
		key_gguf.len = key_len_u32;
		key_gguf.string = new char[key_gguf.len + 1];
		file.read(key_gguf.string, key_gguf.len);
		key_gguf.string[key_gguf.len] = '\0';

		auto value_t_pair = read_gguf_metadata_value(file);
		metadata_map[std::string(key_gguf.string)] = value_t_pair;
		delete[] key_gguf.string;
	}
	weights.config = LlamaConfig::from_metadata(metadata_map);
	

	// Read tensor infos
	std::vector<gguf_tensor_info_t> tensor_infos(header.tensor_count);
	for (uint64_t i = 0; i < header.tensor_count; ++i) {
		gguf_tensor_info_t& info = tensor_infos[i];
		uint32_t name_len_u32;
		file.read(reinterpret_cast<char*>(&name_len_u32), sizeof(uint32_t));
		info.name.len = name_len_u32;
		info.name.string = new char[info.name.len + 1];
		file.read(info.name.string, info.name.len);
		info.name.string[info.name.len] = '\0';
		file.read(reinterpret_cast<char*>(&info.n_dimensions), sizeof(uint32_t));
		if (info.n_dimensions > 4) {
			throw std::runtime_error("Error: Tensor " + std::string(info.name.string) + " has too many dimensions: " + std::to_string(info.n_dimensions) + ". Max 4 dimensions are supported.");
		}
		for (int dim_idx = 0; dim_idx < info.n_dimensions; ++dim_idx) {
			file.read(reinterpret_cast<char*>(&info.dimensions[dim_idx]), sizeof(uint64_t));
		}
		file.read(reinterpret_cast<char*>(&info.type), sizeof(ggml_type));
		file.read(reinterpret_cast<char*>(&info.offset), sizeof(uint64_t));
	}
	
	// load tensors
	for (const auto& tensor_info : tensor_infos) {
		if (tensor_info.type != GGML_TYPE_F32) {
			std::cerr << "Error: Unsupported tensor type: " << tensor_info.type << " for tensor " << tensor_info.name.string << ". Only F32 is supported." << std::endl;
			exit(1);
		}

		std::vector<int> shape(tensor_info.n_dimensions);
		size_t num_elements = 1;
		for (int i = 0; i < tensor_info.n_dimensions; ++i) {
			shape[i] = tensor_info.dimensions[i];
			num_elements *= shape[i];
		}

		Tensor tensor(shape);
		std::vector<float> cpu_data(num_elements);

		// Seek to tensor data offset
		file.seekg(sizeof(gguf_header_t) + header.metadata_kv_count * sizeof(gguf_metadata_kv_t) + sizeof(gguf_tensor_info_t) * header.tensor_count + tensor_info.offset, std::ios::beg);

		if (tensor_info.type == GGML_TYPE_F16) {
			std::vector<uint16_t> cpu_data_f16(num_elements);
			file.read(reinterpret_cast<char*>(cpu_data_f16.data()), cpu_data_f16.size() * sizeof(uint16_t));
			for (size_t i = 0; i < num_elements; ++i) {
				__half h_val;
				h_val = cpu_data_f16[i]; // load raw bits
				cpu_data[i] = __half2float(h_val);
			}
		} else {
			file.read(reinterpret_cast<char*>(cpu_data.data()), cpu_data.size() * sizeof(float));
		}

		CHECK_CUDA(cudaMemcpy(tensor.gpu_data.data, cpu_data.data(), cpu_data.size() * sizeof(float), cudaMemcpyHostToDevice));

		std::string tensor_name(tensor_info.name.string, tensor_info.name.len);
		weights.tensors.emplace(tensor_name, std::move(tensor));
		delete[] tensor_info.name.string; // Free allocated name string
	}

	for (auto const& [key, value] : metadata_map) {
		free_gguf_metadata_value(const_cast<std::pair<gguf_metadata_value_t, gguf_metadata_value_type>&>(value));
	}

	return weights;
}

int main() {
	cublasHandle_t handle;
	cublasCreate(&handle);

	std::string model_path = "llama3-8b.gguf";
	ModelWeights weights = load_weights(model_path);
	LlamaConfig config = weights.config;

	if (config.hidden_size == 0) {
		std::cerr << "Error: Failed to load model configuration. Please check the GGUF file or metadata parsing." << std::endl;
		return 1;
	}

	std::cout << "Model Config:" << std::endl;
	std::cout << "  Vocab Size: " << config.vocab_size << std::endl;
	std::cout << "  Hidden Size: " << config.hidden_size << std::endl;
	std::cout << "  Intermediate Size: " << config.intermediate_size << std::endl;
	std::cout << "  Num Layers: " << config.num_hidden_layers << std::endl;
	std::cout << "  Num Attention Heads: " << config.num_attention_heads << std::endl;
	std::cout << "  Num Key-Value Heads: " << config.num_key_value_heads << std::endl;
	std::cout << "  RMS Norm Epsilon: " << std::fixed << std::setprecision(10) << config.rms_norm_eps << std::endl;
	std::cout << "  Max Pos Embeddings: " << config.max_position_embedding << std::endl;
	std::cout << std::endl;

	LlamaModel model(config, handle);

	// initialize model weights (copy from loaded tensors)
	auto copy_tensor = [&](const std::string& name, Matrix& dest) {
		if (weights.tensors.count(name)) {
			const auto& tensor = weights.tensors.at(name);
			if (dest.rows * dest.cols != tensor.num_elements()) {
				throw std::runtime_error("Tensor size mismatch for " + name);
			}
			CHECK_CUDA(cudaMemcpy(dest.data(), tensor.gpu_data.data, tensor.num_elements() * sizeof(float), cudaMemcpyDeviceToDevice));
		} else {
			throw std::runtime_error("Tensor not found: " + name);
		}
	};

	copy_tensor("model.embed_tokens.weight", model.embedding);
	copy_tensor("model.norm.weight", model.final_norm.gamma);
	copy_tensor("lm_head.weight", model.lm_head);

	for (int layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
		auto prefix = "model.layers." + std::to_string(layer_idx) + ".";
		copy_tensor(prefix + "self_attn.q_proj.weight", model.layers[layer_idx].attn.q_proj);
		copy_tensor(prefix + "self_attn.k_proj.weight", model.layers[layer_idx].attn.k_proj);
		copy_tensor(prefix + "self_attn.v_proj.weight", model.layers[layer_idx].attn.v_proj);
		copy_tensor(prefix + "self_attn.o_proj.weight", model.layers[layer_idx].attn.o_proj);
		copy_tensor(prefix + "mlp.gate_proj.weight", model.layers[layer_idx].ff.gate_proj);
		copy_tensor(prefix + "mlp.up_proj.weight", model.layers[layer_idx].ff.up_proj);
		copy_tensor(prefix + "mlp.down_proj.weight", model.layers[layer_idx].ff.down_proj);
		copy_tensor(prefix + "input_layernorm.weight", model.layers[layer_idx].input_norm.gamma);
		copy_tensor(prefix + "post_attention_layernorm.weight", model.layers[layer_idx].post_attn_norm.gamma);
	}

	// example tokens
	// TODO : use the tokenizer to extract tokens from the text
	std::vector<int> tokens = {1, 2543, 532};
	Matrix logits(1, config.vocab_size, handle);

	model.forward(tokens, logits);

	// get logits result back to CPU for further processing
	std::vector<float> cpu_logits(logits.rows * logits.cols);
	CHECK_CUDA(cudaMemcpy(cpu_logits.data(), logits.data(), logits.rows * logits.cols * sizeof(float), cudaMemcpyDeviceToHost));

	// basic output
	std::vector<std::pair<float, int>> logits_with_index(config.vocab_size);
	for (int i = 0; i < config.vocab_size; ++i) {
		logits_with_index[i] = {cpu_logits[i], i};
	}
	std::sort(logits_with_index.rbegin(), logits_with_index.rend());

	std::cout << "Top 10 Logits:\n";
	for(int i=0; i<10; ++i) {
		std::cout << "Token ID: " << logits_with_index[i].second << ", Logit: " << logits_with_index[i].first << std::endl;
	}

	cublasDestroy(handle);
	cudaDeviceReset();
	return 0;

}