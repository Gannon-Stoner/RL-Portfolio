//
// Created by grs on 1/14/25.
//

#ifndef RLAGENT_H
#define RLAGENT_H

// RLAgent.h
#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <stdexcept>
#include "../environment/PortfolioEnv.h"
#include "../memory/ReplayBuffer.h"

class RLAgent {
private:
    // Neural Network Architecture
    struct Network {
        Eigen::MatrixXd W1, W2, W3;  // Weight matrices
        Eigen::VectorXd b1, b2, b3;  // Bias vectors

        // Constructor
        Network(size_t input_dim, size_t hidden_dim, size_t output_dim);

        // Forward pass through the network
        Eigen::VectorXd forward(const Eigen::VectorXd& state) const;

        // Update weights using gradients
        void update(const std::vector<Eigen::MatrixXd>& weight_gradients,
                   const std::vector<Eigen::VectorXd>& bias_gradients,
                   double learning_rate);
    };

    // Performance Metrics Tracking
    struct Metrics {
        std::vector<double> episode_returns;
        std::vector<double> td_errors;
        std::vector<double> portfolio_values;
        std::vector<double> sharpe_ratios;
        std::vector<double> max_drawdowns;
        std::vector<double> training_losses;

        void clear() {
            episode_returns.clear();
            td_errors.clear();
            portfolio_values.clear();
            sharpe_ratios.clear();
            max_drawdowns.clear();
            training_losses.clear();
        }
    };

    // Network components
    std::unique_ptr<Network> policy_network_;
    std::unique_ptr<Network> target_network_;
    std::shared_ptr<ReplayBuffer> replay_buffer_;

    // Training parameters
    double initial_learning_rate_;
    double learning_rate_;
    double min_learning_rate_;
    double learning_rate_decay_;
    double gamma_;           // Discount factor
    double epsilon_;         // Exploration rate
    double min_epsilon_;     // Minimum exploration rate
    double epsilon_decay_;   // Exploration decay rate
    double tau_;            // Target network update rate
    double max_grad_norm_;   // For gradient clipping

    // Training state
    size_t training_steps_;
    double cumulative_reward_;
    Metrics metrics_;
    std::mt19937 rng_;      // Random number generator

    // Internal methods
    void updateTargetNetwork();
    std::vector<double> computeTDError(const std::vector<Experience>& batch);
    void clipGradients(std::vector<Eigen::MatrixXd>& weight_gradients,
                      std::vector<Eigen::VectorXd>& bias_gradients);
    void updateLearningRate();
    void updateEpsilon();
    void updateMetrics(const Experience& experience);
    void validateState(const Eigen::VectorXd& state) const;

public:
    RLAgent(size_t state_dim,
            size_t action_dim,
            std::shared_ptr<ReplayBuffer> replay_buffer,
            double learning_rate = 0.001,
            double gamma = 0.99,
            double epsilon = 0.1,
            double tau = 0.01);

    // Core RL methods
    TradingAction selectAction(const Eigen::VectorXd& state);
    void train(size_t batch_size);
    void update(const Experience& experience);

    // Utility methods
    void saveModel(const std::string& path) const;
    void loadModel(const std::string& path);
    void setEpsilon(double epsilon) { epsilon_ = epsilon; }
    double getEpsilon() const { return epsilon_; }
    void logMetrics() const;

    // Performance evaluation methods
    [[nodiscard]] double getAverageReturn(size_t window_size = 100) const;
    [[nodiscard]] double getAverageSharpeRatio(size_t window_size = 100) const;
    [[nodiscard]] double getAverageDrawdown(size_t window_size = 100) const;
    [[nodiscard]] const Metrics& getMetrics() const { return metrics_; }

    // Early stopping check
    bool shouldEarlyStop(size_t patience = 10, double min_improvement = 0.001) const;
};
#endif //RLAGENT_H
