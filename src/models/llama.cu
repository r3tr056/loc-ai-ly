#include <models/llama.hpp>

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


// RMS Normalization
RMSNorm::RMSNorm(int dim, float eps, cublasHandle_t handle) : gamma(1, dim, handle), eps(eps), cublas_handle(handle) {
	// initialize gamma
	float* h_gamma = new float[dim];
	std::fill(h_gamma, h_gamma + dim, 1.0f);
	CHECK_CUDA(cudaMemcpy(gamma.data(), h_gamma, dim * sizeof(float), cudaMemcpyHostToDevice));
	delete[] h_gamma;
}

void RMSNorm::forward(const Matrix& input, Matrix& output) {
	if (output.rows != input.rows || output.cols != input.cols) {
		output = Matrix(input.rows, input.cols, cublas_handle);
	}

	dim3 blockDim(32, 32); // FLAG: Tunable paramater
	dim3 gridDim(
		(input.cols + blockDim.x - 1) / blockDim.x,
		(input.rows + blockDim.y - 1) / blockDim.y
	);

	Kernels::ModelKernels::rmsnorm_kernel(output.data_ptr(), input.data_ptr(), gamma.data(), eps, input.rows, input.cols, input.type);

	// cudaError_t err = cudaGetLastError();
	// if (err != cudaSuccess) {
	//     throw std::runtime_error("CUDA kernel launch error: " + std::string(cudaGetErrorString(err)));
	// }

	CHECK_CUDA(cudaDeviceSynchronize());
}


// Rotary Positional Embedding
RotaryEmbedding::RotaryEmbedding(int dim, int max_seq_len, cublasHandle_t handle) : sin_cached({max_seq_len, dim / 2}), cos_cached({max_seq_len, dim / 2}), cublas_handle(handle)
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

void RotaryEmbedding::apply(Matrix& x, int start_pos) {
	dim3 blockDim(256);
	dim3 gridDim((x.rows + blockDim.x - 1) / blockDim.x);

	Kernels::ModelKernels::rope_kernel(x.data_ptr(), cos_cached.gpu_data.data, sin_cached.gpu_data.data, start_pos, x.rows, x.cols, x.type);
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		throw std::runtime_error("CUDA kernel launch error: " + std::string(cudaGetErrorString(err)));
	}
	CHECK_CUDA(cudaDeviceSynchronize());
}


// Attention Layer with KV Cache
Attention::Attention(const LlamaConfig& config, cublasHandle_t handle)
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

	
void Attention::forward(const Matrix& input, Matrix& output, int start_pos) {
	const int q_dim = config.num_attention_heads * config.head_dim();
	const int kv_dim = config.num_key_value_heads * config.head_dim();

	Matrix q(input.rows, q_dim, cublas_handle, input.type);
	q.gemm(input, q_proj);

	Matrix k(input.rows, kv_dim, cublas_handle, input.type);
	k.gemm(input, k_proj);

	Matrix v(input.rows, kv_dim, cublas_handle, input.type);
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

	// fused attention kernel call
	Matrix attn_output(input.cols, q_dim, cublas_handle, input.type);
	Kernels::ModelKernels::fused_attention_kernel(attn_output.data_ptr(), q.data_ptr(), key_cache.data(), value_cache.data(), input.rows, q_dim, kv_dim, cache_pos, config.head_dim(), config.num_attention_heads, config.num_key_value_heads, config.num_attention_heads / config.num_key_value_heads, input.type);
	CHECK_CUDA(cudaDeviceSynchronize());

	// output projection
	output.gemm(attn_output, o_proj);
}

// Feed forward network (SwiGLU)
FeedForward::FeedForward(const LlamaConfig& config, cublasHandle_t handle)
	: gate_proj({config.hidden_size, config.intermediate_size}, nullptr, handle, true),
	up_proj({config.hidden_size, config.intermediate_size}, nullptr, handle, true),
	down_proj({config.intermediate_size, config.hidden_size}, nullptr, handle, true),
	cublas_handle(handle)
{
	gate_proj = Matrix(config.hidden_size, config.intermediate_size, handle);
	up_proj = Matrix(config.hidden_size, config.intermediate_size, handle);
	down_proj = Matrix(config.intermediate_size, config.hidden_size, handle);
}

void FeedForward::forward(const Matrix& input, Matrix& output) {
	ggml_type activation_type = input.type;

	Matrix gate(input.rows, gate_proj.cols, cublas_handle, input.type);
	gate.gemm(input, gate_proj);

	Matrix up(input.rows, up_proj.cols, cublas_handle, input.type);
	up.gemm(input, up_proj);

	// SwiGLU activation (CUDA kernel)
	dim3 blockDimSwiglu(256); // FLAG: Tunable parameter
	dim3 gridDimSwiglu((gate.rows * gate.cols + blockDimSwiglu.x - 1) / blockDimSwiglu.x);
	Kernels::ModelKernels::swiglu_kernel(gate.data_ptr(), up.data_ptr(), gate.rows * gate.cols, activation_type);

	// Error checking
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error("CUDA kernel launch error: " + std::string(cudaGetErrorString(err)));
	}
	CHECK_CUDA(cudaDeviceSynchronize());
	output.gemm(gate, down_proj);
}

// Transformer block
TransformerBlock::TransformerBlock(const LlamaConfig& config, cublasHandle_t handle)
	: attn(config, handle), ff(config, handle),
	input_norm(config.hidden_size, config.rms_norm_eps, handle),
	post_attn_norm(config.hidden_size, config.rms_norm_eps, handle),
	cublas_handle(handle) {}

void TransformerBlock::forward(const Matrix& input, Matrix& output, int pos) {
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



LlamaModel::LlamaModel(const LlamaConfig& config, cublasHandle_t handle)
		: layers(config.num_hidden_layers, TransformerBlock(config, handle)),
		embedding({config.vocab_size, config.hidden_size}, nullptr, handle, true),
		final_norm(config.hidden_size, config.rms_norm_eps, handle),
		lm_head({config.hidden_size, config.vocab_size}, nullptr, handle, true),
		cublas_handle(handle), llama_config(config)
{
	embedding = Matrix(config.vocab_size, config.hidden_size, handle);
	lm_head = Matrix(config.hidden_size, config.vocab_size, handle);
}

void LlamaModel::forward(const std::vector<int>& tokens, Matrix& logits) {
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

// Llama3 family Model Weight Loading method
LlamaModelWeights load_weights(const std::string& path) {
	std::ifstream file(path, std::ios::binary);
	LlamaModelWeights weights;
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
		ggml_type tensor_type_to_load = GGML_TYPE_F32;
		
		if (tensor_info.type == GGML_TYPE_F16) {
			tensor_type_to_load = GGML_TYPE_F16;
		} else {
			tensor_type_to_load = GGML_TYPE_F32;
		}

		std::vector<int> shape(tensor_info.n_dimensions);
		size_t num_elements = 1;
		for (int i = 0; i < tensor_info.n_dimensions; ++i) {
			shape[i] = tensor_info.dimensions[i];
			num_elements *= shape[i];
		}

		Tensor tensor(shape, tensor_info.type);
		std::vector<float> cpu_data(num_elements);

		// Seek to tensor data offset
		file.seekg(sizeof(gguf_header_t) + header.metadata_kv_count * sizeof(gguf_metadata_kv_t) + sizeof(gguf_tensor_info_t) * header.tensor_count + tensor_info.offset, std::ios::beg);

		if (tensor_info.type == GGML_TYPE_F16 && tensor_type_to_load == GGML_TYPE_F16) {
			// Load FP16 as FP16
			std::vector<uint16_t> cpu_data_f16(num_elements);
			file.read(reinterpret_cast<char*>(cpu_data_f16.data()), cpu_data_f16.size() * sizeof(half));

			CHECK_CUDA(cudaMalloc(&tensor.gpu_data.data, num_elements * sizeof(half)));
			CHECK_CUDA(cudaMemcpy(tensor.gpu_data.data, cpu_data_f16.data(), cpu_data_f16.size() * sizeof(half), cudaMemcpyHostToDevice));

		} else if (tensor_info.type == GGML_TYPE_I8) {
			std::vector<int8_t> cpu_data_int8(num_elements);
			file.read(reinterpret_cast<char*>(cpu_data_int8.data()), cpu_data_int8.size() * sizeof(int8_t));
			std::vector<float> cpu_data_f32 = dequantize_int8_to_f32(cpu_data_int8);
			cpu_data = cpu_data_f32;

			CHECK_CUDA(cudaMalloc(&tensor.gpu_data.data, num_elements * sizeof(half)));
			CHECK_CUDA(cudaMemcpy(tensor.gpu_data.data, cpu_data_f32.data(), cpu_data_f32.size() * sizeof(float), cudaMemcpyHostToDevice));

		} else {
			// Load F32 (for F32 or dequantized F16 to F32 if needed)
			std::vector<float> cpu_data_f32(num_elements);
			if (tensor_info.type == GGML_TYPE_F16) {
				// Load F16 data from file and convert to F32 on host first
				std::vector<uint16_t> cpu_data_f16_raw(num_elements);
				file.read(reinterpret_cast<char*>(cpu_data_f16_raw.data()), cpu_data_f16_raw.size() * sizeof(uint16_t));
				std::vector<float> cpu_data_f32_converted = half_to_float_host(std::vector<half>((half*)cpu_data_f16_raw.data(), (half*)cpu_data_f16_raw.data() + num_elements)); // Host side conversion
				cpu_data_f32 = cpu_data_f32_converted;
			} else {
				// Load F32 directly
				file.read(reinterpret_cast<char*>(cpu_data_f32.data()), cpu_data_f32.size() * sizeof(float));
			}

			CHECK_CUDA(cudaMalloc(&tensor.gpu_data.data, num_elements * sizeof(float)));
			CHECK_CUDA(cudaMemcpy(tensor.gpu_data.data, cpu_data_f32.data(), cpu_data_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
		}

		std::string tensor_name(tensor_info.name.string, tensor_info.name.len);
		weights.tensors.emplace(tensor_name, std::move(tensor));
		delete[] tensor_info.name.string; // Free allocated name string
	}

	for (auto const& [key, value] : metadata_map) {
		free_gguf_metadata_value(const_cast<std::pair<gguf_metadata_value_t, gguf_metadata_value_type>&>(value));
	}

	return weights;
};
