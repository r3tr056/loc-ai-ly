#include <model_manager/model_manager.hpp>
#include <filesystem>

namespace fs = std::filesystem;

ModelManager::ModelManager() {
    cublasCreate(&cublasHandle);
    loadModelsFromStore("models");
}

ModelManager::~ModelManager() {
    cublasDestroy(cublasHandle);
}

bool ModelManager::setModelQuantization(const std::string& modelName, ggml_type quantizationType) {
    if (!loadedModels.count(modelName)) {
        return false;
    }
    modelQuantizationTypes[modelName] = quantizationType;
    // TODO : reload or re-quantize weights here if switching dynamically.
    return true;
}

ggml_type ModelManager::getModelQuantization(const std::string& modelName) const {
    if (modelQuantizationTypes.count(modelName)) {
        return modelQuantizationTypes.at(modelName);
    }
    return GGML_TYPE_F32;  // default to f32 if not type is set
}

bool ModelManager::loadModel(const std::string& modelName, const std::string& modelPath) {
    try {
        LlamaModelWeights weights = load_weights(modelPath);
        LlamaConfig config = weights.config;
        ggml_type loaded_type = GGML_TYPE_F32;

        if (config.hidden_size == 0) {
            std::cerr << "Error: Failed to load model configuration. Please check the GGUF file or metadata parsing." << std::endl;
            return false;
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

        if (weights.tensors.begin()->second.type == GGML_TYPE_F16) loaded_type = GGML_TYPE_F16;
        std::unique_ptr<LlamaModel> model = std::make_unique<LlamaModel>(config, cublasHandle);

        auto copy_tensor = [&](const std::string& name, Matrix& dest) {
            if (weights.tensors.count(name)) {
                const auto& tensor = weights.tensors.at(name);
                if (dest.rows * dest.cols != tensor.num_elements()) {
                    throw std::runtime_error("Tensor size mismatch for " + name + ". Expected: " + std::to_string(dest.rows * dest.cols) + ", Got: " + std::to_string(tensor.num_elements()));
                }
                CHECK_CUDA(cudaMemcpy(dest.data(), tensor.gpu_data.data, tensor.num_elements() * sizeof(float), cudaMemcpyDeviceToDevice));
            } else {
                throw std::runtime_error("Tensor not found: " + name);
            }
        };

        copy_tensor("model_embed_tokens_weight", model->embedding);
        copy_tensor("model_norm_weight", model->final_norm.gamma);
        copy_tensor("lm_head_weight", model->lm_head);

        for (int layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
            auto prefix = "model_layers_" + std::to_string(layer_idx) + "_";
            copy_tensor(prefix + "input_layernorm_weight", model->layers[layer_idx].input_norm.gamma);
            copy_tensor(prefix + "post_attention_layernorm_weight", model->layers[layer_idx].post_attn_norm.gamma);
            copy_tensor(prefix + "self_attn_q_proj_weight", model->layers[layer_idx].attn.q_proj);
            copy_tensor(prefix + "self_attn_k_proj_weight", model->layers[layer_idx].attn.k_proj);
            copy_tensor(prefix + "self_attn_v_proj_weight", model->layers[layer_idx].attn.v_proj);
            copy_tensor(prefix + "self_attn_o_proj_weight", model->layers[layer_idx].attn.o_proj);
            copy_tensor(prefix + "mlp_gate_proj_weight", model->layers[layer_idx].ff.gate_proj);
            copy_tensor(prefix + "mlp_up_proj_weight", model->layers[layer_idx].ff.up_proj);
            copy_tensor(prefix + "mlp_down_proj_weight", model->layers[layer_idx].ff.down_proj);
        }

        std::string tokenizerModelPath = fs::path(modelPath).parent_path() / (fs::path(modelPath).stem().string() + ".tokenizer.model");
        if (!fs::exists(tokenizerModelPath)) {
            std::cerr << "Warning: Tokenizer model not found at: " << tokenizerModelPath << " for model: " << modelName << ". Text input will not be tokenized properly." << std::endl;
        } else {
            try {
                std::unique_ptr<Tokenizer> tokenizer = std::make_unique<Tokenizer>(tokenizerModelPath);
                modelTokenizers[modelName] = std::move(tokenizer);
                std::cout << "Tokenizer model loaded for: " << modelName << " from: " << tokenizerModelPath << std::endl;
            } catch (const std::exception& tokenizer_ex) {
                std::cerr << "Error loading tokenizer for model '" << modelName << "' from " << tokenizerModelPath << ": " << tokenizer_ex.what() << std::endl;
                std::cerr << "Text input will not be tokenized properly." << std::endl;
            }
        }

        loadedModels[modelName] = std::move(model);
        modelQuantizationTypes[modelName] = loaded_type;
        std::cout << "Model '" << modelName << "' loaded successfully from " << modelPath << " as type: " << loaded_type << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error loading model '" << modelName << "' from " << modelPath << ": " << e.what() << std::endl;
        return false;
    }
}

LlamaModel* ModelManager::getModel(const std::string& modelName) {
    if (loadedModels.count(modelName)) {
        return loadedModels[modelName].get();
    }
    return nullptr;
}

void ModelManager::loadModelsFromStore(const std::string& storePath) {
    if (!fs::exists(storePath) || !fs::is_directory(storePath)) {
        std::cerr << "Warning: Model store directory '" << storePath << "' not found or not a directory." << std::endl;
        return;
    }

    for (const auto& entry : fs::directory_iterator(storePath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
            std::string modelName = entry.path().stem().string();
            std::string modelFullPath = entry.path().string();
            loadModel(modelName, modelFullPath);
        }
    }
}

Tokenizer* ModelManager::getTokenizer(const std::string& modelName) {
    if (modelTokenizers.count(modelName)) {
        return modelTokenizers[modelName].get();
    }
    return nullptr;
}