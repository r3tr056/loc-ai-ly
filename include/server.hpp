#ifndef SERVER_H
#define SERVER_H

#include <cpp_httplib.h>
#include "nlohmann/json.hpp"
#include <model_manager/model_manager.hpp>

void handle_chat_completions(const httplib::Request& req, httplib::Response& res, ModelManager& modelManager);

#endif // SERVER_H