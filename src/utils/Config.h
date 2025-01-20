//
// Created by grs on 1/17/25.
//

#ifndef CONFIG_H
#define CONFIG_H

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

class Config {
public:
    struct TradingParams {
        double initial_capital = 100000.0;
        double transaction_cost = 0.001;
        double max_position_size = 0.5;
        size_t batch_size = 64;
        size_t num_episodes = 1000;
        double learning_rate = 0.001;
        double gamma = 0.99;
        double epsilon = 0.1;
        double min_epsilon = 0.01;
        std::vector<std::string> symbols;
    };

    static TradingParams loadConfig(const std::string& config_path) {
        TradingParams params;
        try {
            std::ifstream f(config_path);
            if (!f.is_open()) {
                spdlog::warn("Config file not found, using default parameters");
                return params;
            }

            nlohmann::json data = nlohmann::json::parse(f);

            params.initial_capital = data.value("initial_capital", params.initial_capital);
            params.transaction_cost = data.value("transaction_cost", params.transaction_cost);
            params.max_position_size = data.value("max_position_size", params.max_position_size);
            params.batch_size = data.value("batch_size", params.batch_size);
            params.num_episodes = data.value("num_episodes", params.num_episodes);
            params.learning_rate = data.value("learning_rate", params.learning_rate);
            params.gamma = data.value("gamma", params.gamma);
            params.epsilon = data.value("epsilon", params.epsilon);
            params.min_epsilon = data.value("min_epsilon", params.min_epsilon);
            params.symbols = data.value("symbols", std::vector<std::string>{"AAPL", "MSFT", "GOOGL"});

        } catch (const std::exception& e) {
            spdlog::error("Error loading config: {}", e.what());
            spdlog::info("Using default parameters");
        }
        return params;
    }
};
#endif //CONFIG_H
