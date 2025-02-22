#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <sentencepiece_processor.h>

class Tokenizer {
public:
    Tokenizer(const std::string& model_path);
    ~Tokenizer();

    std::vector<int> tokenize(const std::string& text) const;
    std::string detokenize(const std::vector<int>& tokens) const;

private:
    sentencepiece::SentencePieceProcessor processor;
}

#endif // TOKENIZER_H