
#include <model_manager/tokenizer.hpp>
#include <stdexcept>

Tokenizer::Tokenizer(const std::string& model_path) {
    sentencepiece::util::Status status = processor.Load(model_path);
    if (!status.ok()) {
        throw std::runtime_error("Failed to load SentencePiece model from: " + model_path + ". Error: " + status.ToString());
    }
}

Tokenizer::~Tokenizer() {}

std::vector<int> Tokenizer::tokenize(const std::string& text) const {
    std::vector<int> token_ids;
    processor.Encode(text, &token_ids);
    return token_ids;
}

std::string Tokenizer::detokenize(const std::vector<int>& tokens) const {
    std::string text;
    processor.Decode(tokens, &text);
    return text;
}