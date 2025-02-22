#ifndef LLAMA_MODEL_H
#define LLAMA_MODEL_H

#include <map>
#include <unordered_map>
#include <inference_engine.hpp>
#include <model_manager/gguf_loader.hpp>

// Llama3 8B config
struct LlamaConfig : public Config {
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

	std::string model_type() const override { return "llama"; }
};

struct LlamaModelWeights {
	std::unordered_map<std::string, Tensor> tensors;
	LlamaConfig config;
};

LlamaModelWeights load_weights(const std::string& path);


// layers
class RMSNorm {
public:
	Matrix gamma;
	float eps;
	cublasHandle_t cublas_handle;

	RMSNorm(int dim, float eps, cublasHandle_t handle);
	void forward(const Matrix& input, Matrix& output);

};

class RotaryEmbedding {
public:
	std::vector<float> inv_freq;
	Tensor sin_cached;
	Tensor cos_cached;
	cublasHandle_t cublas_handle;

	RotaryEmbedding(int dim, int max_seq_len, cublasHandle_t handle);
	void apply(Matrix& x, int start_pos);
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

	Attention(const LlamaConfig& config, cublasHandle_t handle);
	void forward(const Matrix& input, Matrix& output, int start_pos);
};

// feed forward network (SwiGLU)
class FeedForward {
public:
	Matrix gate_proj, up_proj, down_proj;
	cublasHandle_t cublas_handle;

	FeedForward(const LlamaConfig& config, cublasHandle_t handle);
	void forward(const Matrix& input, Matrix& output);
};

class TransformerBlock {
public:
	Attention attn;
	FeedForward ff;
	RMSNorm input_norm, post_attn_norm;
	cublasHandle_t cublas_handle;
public:
    TransformerBlock(const LlamaConfig& config, cublasHandle_t handle);
    void forward(const Matrix& input, Matrix& output, int pos);
};

// LlamaModel class (declaration - implementation will be in llama_model.cu)
class LlamaModel : public Model {
public:
	std::vector<TransformerBlock> layers;
	Matrix embedding;
	RMSNorm final_norm;
	Matrix lm_head;
	int current_pos = 0;
	cublasHandle_t cublas_handle;
	LlamaConfig llama_config;

public:
    LlamaModel(const LlamaConfig& config, cublasHandle_t handle);
   	void forward(const std::vector<int>& tokens, Matrix& logits) override;
	LlamaConfig& get_config() { return llama_config; } 
};



#endif // LLAMA_MODEL_H