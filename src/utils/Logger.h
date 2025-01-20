//
// Created by grs on 1/14/25.
//

#ifndef LOGGER_H
#define LOGGER_H

#pragma once

#include <string>
#include <fstream>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/fmt/ostr.h>
#include <fmt/format.h>
#include <iostream>

class Logger {
public:
    enum class LogLevel {
        DEBUG,
        INFO,
        WARNING,
        ERROR
    };

    Logger(const std::string& log_path, LogLevel level = LogLevel::INFO);

    // Basic logging methods
    void debug(const std::string& message);
    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);

    // Formatted logging methods
    template<typename... Args>
    void debug_fmt(fmt::format_string<Args...> fmt, Args&&... args) {
        try {
            logger_->debug(fmt::format(fmt, std::forward<Args>(args)...));
        } catch (const std::exception& e) {
            std::cerr << "Error in debug logging: " << e.what() << std::endl;
        }
    }

    template<typename... Args>
    void info_fmt(fmt::format_string<Args...> fmt, Args&&... args) {
        try {
            logger_->info(fmt::format(fmt, std::forward<Args>(args)...));
        } catch (const std::exception& e) {
            std::cerr << "Error in info logging: " << e.what() << std::endl;
        }
    }

    template<typename... Args>
    void warning_fmt(fmt::format_string<Args...> fmt, Args&&... args) {
        try {
            logger_->warn(fmt::format(fmt, std::forward<Args>(args)...));
        } catch (const std::exception& e) {
            std::cerr << "Error in warning logging: " << e.what() << std::endl;
        }
    }

    template<typename... Args>
    void error_fmt(fmt::format_string<Args...> fmt, Args&&... args) {
        try {
            logger_->error(fmt::format(fmt, std::forward<Args>(args)...));
        } catch (const std::exception& e) {
            std::cerr << "Error in error logging: " << e.what() << std::endl;
        }
    }

    // Performance logging
    void logTrainingMetrics(const std::string& metrics);
    void logBacktestResults(const std::string& results);
    void logPortfolioState(const std::string& state);

    // Utility methods
    void setLogLevel(LogLevel level);
    void flush();

private:
    std::shared_ptr<spdlog::logger> logger_;
    std::ofstream metrics_file_;
    LogLevel current_level_;
};

#endif //LOGGER_H