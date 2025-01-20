//
// Created by grs on 1/14/25.
//

#include "Logger.h"
#include <filesystem>
#include <iostream>

Logger::Logger(const std::string& log_path, LogLevel level)
    : current_level_(level) {
    try {
        // Create directories if they don't exist
        std::filesystem::path log_dir = std::filesystem::path(log_path).parent_path();
        if (!log_dir.empty()) {
            std::filesystem::create_directories(log_dir);
        }

        // Initialize spdlog logger
        logger_ = spdlog::basic_logger_mt("portfolio_logger", log_path);

        // Set logger pattern: [timestamp] [level] message
        logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

        // Set initial log level
        setLogLevel(level);

        // Open metrics file
        std::string metrics_path = log_dir / "metrics.csv";
        metrics_file_.open(metrics_path, std::ios::app);
        if (!metrics_file_.is_open()) {
            throw std::runtime_error("Failed to open metrics file: " + metrics_path);
        }

        // Write metrics header if file is empty
        if (metrics_file_.tellp() == 0) {
            metrics_file_ << "Timestamp,Type,Value,Additional_Info\n";
        }

        info("Logger initialized successfully");

    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
        throw;
    }
}

void Logger::debug(const std::string& message) {
    try {
        logger_->debug(message);
    } catch (const std::exception& e) {
        std::cerr << "Error in debug logging: " << e.what() << std::endl;
    }
}

void Logger::info(const std::string& message) {
    try {
        logger_->info(message);
    } catch (const std::exception& e) {
        std::cerr << "Error in info logging: " << e.what() << std::endl;
    }
}

void Logger::warning(const std::string& message) {
    try {
        logger_->warn(message);
    } catch (const std::exception& e) {
        std::cerr << "Error in warning logging: " << e.what() << std::endl;
    }
}

void Logger::error(const std::string& message) {
    try {
        logger_->error(message);
    } catch (const std::exception& e) {
        std::cerr << "Error in error logging: " << e.what() << std::endl;
    }
}

void Logger::logTrainingMetrics(const std::string& metrics) {
    try {
        // Log to file logger
        logger_->info("Training Metrics: {}", metrics);

        // Log to metrics file with timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        metrics_file_ << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S")
                     << ",Training," << metrics << std::endl;
    } catch (const std::exception& e) {
        error("Failed to log training metrics: " + std::string(e.what()));
    }
}

void Logger::logBacktestResults(const std::string& results) {
    try {
        // Log to file logger
        logger_->info("Backtest Results: {}", results);

        // Log to metrics file with timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        metrics_file_ << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S")
                     << ",Backtest," << results << std::endl;
    } catch (const std::exception& e) {
        error("Failed to log backtest results: " + std::string(e.what()));
    }
}

void Logger::logPortfolioState(const std::string& state) {
    try {
        // Log to file logger
        logger_->info("Portfolio State: {}", state);

        // Log to metrics file with timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        metrics_file_ << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S")
                     << ",Portfolio," << state << std::endl;
    } catch (const std::exception& e) {
        error("Failed to log portfolio state: " + std::string(e.what()));
    }
}

void Logger::setLogLevel(LogLevel level) {
    try {
        current_level_ = level;
        spdlog::level::level_enum spdlog_level;

        switch (level) {
            case LogLevel::DEBUG:
                spdlog_level = spdlog::level::debug;
                break;
            case LogLevel::INFO:
                spdlog_level = spdlog::level::info;
                break;
            case LogLevel::WARNING:
                spdlog_level = spdlog::level::warn;
                break;
            case LogLevel::ERROR:
                spdlog_level = spdlog::level::err;
                break;
            default:
                spdlog_level = spdlog::level::info;
        }

        logger_->set_level(spdlog_level);
        logger_->info("Log level set to: {}", spdlog::level::to_string_view(spdlog_level));

    } catch (const std::exception& e) {
        std::cerr << "Error setting log level: " << e.what() << std::endl;
    }
}

void Logger::flush() {
    try {
        logger_->flush();
        metrics_file_.flush();
    } catch (const std::exception& e) {
        std::cerr << "Error flushing logs: " << e.what() << std::endl;
    }
}