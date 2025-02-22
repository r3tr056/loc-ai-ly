
#include <server.hpp>
#include <model_manager/model_manager.hpp>



using json = nlohmann::json;

void handle_chat_completions(const httplib::Request& req, httplib::Response& res, ModelManager& modelManager) {
	try {
		json request_json = json::parse(req.body);
		std::string model_name = request_json.value("model", "");
		std::vector<int> tokens; // Tokenized prompt

		if (model_name.empty()) {
			res.status = 400;
			res.set_content("{\"error\": \"Model name is required.\"}", "application/json");
			return;
		}

		LlamaModel* model = modelManager.getModel(model_name);
		Tokenizer* tokenizer = modelManager.getTokenizer(model_name);
		if (!model) {
			res.status = 404;
			res.set_content("{\"error\": \"Model not found: " + model_name + "\"}", "application/json");
			return;
		}
		if (!tokenizer) {
            res.status = 500;
            res.set_content("{\"error\": \"Tokenizer not loaded for model: " + model_name + ". Cannot process text input.\"}", "application/json");
            return;
        }

		// --- Dummy Tokenization for now - Replace with actual tokenizer ---
		std::string prompt_text;
		if (request_json.contains("messages") && request_json["messages"].is_array()) {
			for (const auto& message : request_json["messages"]) {
				if (message.is_object() && message.value("role", "") == "user") {
					prompt_text += message.value("content", "");
				}
			}
		} else {
			res.status = 400;
			res.set_content("{\"error\": \"Invalid or missing 'messages' array in request.\"}", "application/json");
			return;
		}

		tokens = tokenizer->tokenize(prompt_text);
		if (tokens.empty()) tokens = {1, 2543, 532};

		cublasHandle_t handle; // Get handle - Ideally reuse from ModelManager or thread-local
		cublasCreate(&handle);
		Matrix logits(1, model->get_config().vocab_size, handle);
		model->forward(tokens, logits);
		cublasDestroy(handle);

		// --- Get Top Logit (Dummy sampling) ---
		std::vector<float> cpu_logits(logits.rows * logits.cols);
		CHECK_CUDA(cudaMemcpy(cpu_logits.data(), logits.data(), logits.rows * logits.cols * sizeof(float), cudaMemcpyDeviceToHost));
		std::vector<std::pair<float, int>> logits_with_index(model->get_config().vocab_size);
		for(int i=0; i<model->get_config().vocab_size; ++i) {
			logits_with_index[i] = {cpu_logits[i], i};
		}
		std::sort(logits_with_index.rbegin(), logits_with_index.rend());
		int generated_token_id = logits_with_index[0].second; // Top token
		
		std::string generated_text = tokenizer->detokenize({generated_token_id});

		// --- Format OpenAI API Response ---
		json response_json;
		response_json["id"] = "cmpl-" + std::to_string(std::rand()); // Dummy completion ID
		response_json["object"] = "chat.completion";
		response_json["created"] = std::time(0);
		response_json["model"] = model_name;
		response_json["choices"] = json::array();
		json choice;
		choice["index"] = 0;
		choice["message"]["role"] = "assistant";
		choice["message"]["content"] = generated_text;
		choice["finish_reason"] = "stop"; // Or "length" if max tokens reached

		response_json["choices"].push_back(choice);
		response_json["usage"]["prompt_tokens"] = tokens.size(); // Dummy token counts
		response_json["usage"]["completion_tokens"] = 1;
		response_json["usage"]["total_tokens"] = tokens.size() + 1;


		res.set_content(response_json.dump(4), "application/json"); // Pretty print JSON for readability

	} catch (const json::parse_error& err) {
		res.status = 400;
		res.set_content("{\"error\": \"Invalid JSON in request body.\"}", "application/json");
	} catch (const std::exception& e) {
		res.status = 500;
		res.set_content("{\"error\": \"Server error: " + std::string(e.what()) + "\"}", "application/json");
	}
}

void handle_set_quantization(const httplib::Request& req, httplib::Response& res, ModelManager& modelManager) {
	try {
		json request_json = json::parse(req.body);
		std::string model_name = request_json.value("model", "");
		std::string quantization_level_str = request_json.value("quantization_level", "");

		ggml_type quantization_level = GGML_TYPE_F32;
		if (model_name.empty() || quantization_level_str.empty()) {
			res.status = 400;
			res.set_content("{\"error\": \"Model name and quantization_level are required.\"}", "application/json");
			return;
		}

		if ("FP32" == quantization_level_str) quantization_level = GGML_TYPE_F32;
		else if ("FP16" == quantization_level_str) quantization_level = GGML_TYPE_F16;
		else if ("INT8" == quantization_level_str) quantization_level = GGML_TYPE_I8;

		if (!modelManager.setModelQuantization(model_name, quantization_level)) {
			res.status = 404;
			res.set_content("{\"error\": \"Model not found: " + model_name + "\"}", "application/json");
            return;
		}

		res.set_content("{\"status\": \"Quantization level set to " + quantization_level_str + " for model " + model_name + "\"}", "application/json");

	} catch (const json::parse_error& err) {
		res.status = 400;
        res.set_content("{\"error\": \"Invalid JSON in request body.\"}", "application/json");
	} catch (const std::exception& e) {
		res.status = 500;
        res.set_content("{\"error\": \"Server error: " + std::string(e.what()) + "\"}", "application/json");
	}
}

int main() {
    httplib::Server svr;
    ModelManager modelManager; // Initialize Model Manager - loads models at startup

    // Chat Completions Endpoint
    svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        handle_chat_completions(req, res, modelManager);
    });

	svr.Post("/v1/models/quantization", [&](const httplib::Request& req, httplib::Response& res) {
        handle_set_quantization(req, res, modelManager);
    });

    std::cout << "Server listening on http://localhost:8080" << std::endl;
    svr.listen("0.0.0.0", 8080);

    return 0;
}