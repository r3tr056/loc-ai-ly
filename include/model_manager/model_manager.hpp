#ifndef MODEL_MANAGER_H
#define MODEL_MANAGER_H

#include <models/llama.hpp>
#include <model_manager/tokenizer.hpp>
#include <string>
#include <map>

class ModelManager {
public:
    ModelManager();
    ~ModelManager();

    bool setModelQuantization(const std::string& modelName, ggml_type quantizationType);
    ggml_type getModelQuantization(const std::string& modelName) const;

    bool loadModel(const std::string& modelName, const std::string& modelPath);
    LlamaModel* getModel(const std::string& modelName);

    void loadModelsFromStore(const std::string& storePath);
    Tokenizer* getTokenizer(const std::string& modelName);

private:
    std::map<std::string, std::unique_ptr<LlamaModel>> loadedModels;
    std::map<std::string, std::unique_ptr<Tokenizer>> modelTokenizers;
    cublasHandle_t cublasHandle;
    std::map<std::string, ggml_type> modelQuantizationTypes;
};

#endif // MODEL_MANAGER_H