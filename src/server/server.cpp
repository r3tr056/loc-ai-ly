
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

struct ChatCompletionRequest {
    std::string model;
    std::vector<nlohmann::json> messages;
    int max_tokens = 256;
    float temperature = 0.7f;
    int top_p = 1;
    int n = 1;
    bool stream = false;

    // static ChatCompletionRequest from_json(const nlohmann::json& json_req) {
    //     ChatCompletionRequest req;
    //     req.model = json_req.value("model", "");
    //     req.max_tokens = json_req.value("max_tokens", "");
    //     req.temperature = json_req.value("temperature", "");
    // }
};


struct ChatChoice {
    int index;
    nlohmann::json message; // Message object
    std::string finish_reason;
};

struct ChatCompletionUsage {
    int prompt_tokens;
    int completion_tokens;
    int total_tokens;
};

struct ChatCompletionResponse {
    std::string id;
    std::string object = "chat.completion";
    int created; // Timestamp
    std::string model;
    std::vector<ChatChoice> choices;
    ChatCompletionUsage usage;

    nlohmann::json to_json() const;
};