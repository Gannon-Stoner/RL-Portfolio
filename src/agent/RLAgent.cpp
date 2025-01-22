//
// Created by grs on 1/14/25.
//

#include "RLAgent.h"
#include <cmath>
#include <iostream>
#include <algorithm>

// Helper functions
inline Eigen::VectorXd relu(const Eigen::VectorXd& x) {
    return x.array().max(0.0);
}

inline Eigen::VectorXd softmax(const Eigen::VectorXd& x) {
    Eigen::VectorXd exp_x = x.array().exp();
    return exp_x.array() / exp_x.sum();
}

// Network implementation
RLAgent::Network::Network(size_t input_dim, size_t hidden_dim, size_t output_dim) {
    // Xavier initialization
    double w1_bound = sqrt(6.0 / (input_dim + hidden_dim));
    double w2_bound = sqrt(6.0 / (hidden_dim + hidden_dim));
    double w3_bound = sqrt(6.0 / (hidden_dim + output_dim));

    W1 = Eigen::MatrixXd::Random(hidden_dim, input_dim) * w1_bound;
    W2 = Eigen::MatrixXd::Random(hidden_dim, hidden_dim) * w2_bound;
    W3 = Eigen::MatrixXd::Random(output_dim, hidden_dim) * w3_bound;

    b1 = Eigen::VectorXd::Zero(hidden_dim);
    b2 = Eigen::VectorXd::Zero(hidden_dim);
    b3 = Eigen::VectorXd::Zero(output_dim);
}

Eigen::VectorXd RLAgent::Network::forward(const Eigen::VectorXd& state) const {
    Eigen::VectorXd h1 = relu((W1 * state) + b1);
    Eigen::VectorXd h2 = relu((W2 * h1) + b2);
    Eigen::VectorXd output = W3 * h2 + b3;
    return softmax(output);
}

void RLAgent::Network::update(const std::vector<Eigen::MatrixXd>& weight_gradients,
                            const std::vector<Eigen::VectorXd>& bias_gradients,
                            double learning_rate) {
    W1 -= learning_rate * weight_gradients[0];
    W2 -= learning_rate * weight_gradients[1];
    W3 -= learning_rate * weight_gradients[2];

    b1 -= learning_rate * bias_gradients[0];
    b2 -= learning_rate * bias_gradients[1];
    b3 -= learning_rate * bias_gradients[2];
}

// RLAgent implementation
RLAgent::RLAgent(size_t state_dim, size_t action_dim,
                 std::shared_ptr<ReplayBuffer> replay_buffer,
                 double learning_rate, double gamma,
                 double epsilon, double tau)
    : replay_buffer_(replay_buffer)
    , initial_learning_rate_(learning_rate)
    , learning_rate_(learning_rate)
    , min_learning_rate_(0.00001)
    , learning_rate_decay_(0.9999)
    , gamma_(gamma)
    , epsilon_(epsilon)
    , min_epsilon_(0.01)
    , epsilon_decay_(0.995)
    , tau_(tau)
    , max_grad_norm_(1.0)
    , training_steps_(0)
    , cumulative_reward_(0.0)
    , rng_(std::random_device{}()) {

    constexpr size_t HIDDEN_DIM = 128;
    policy_network_ = std::make_unique<Network>(state_dim, HIDDEN_DIM, action_dim);
    target_network_ = std::make_unique<Network>(state_dim, HIDDEN_DIM, action_dim);
    updateTargetNetwork();
}


void RLAgent::validateState(const Eigen::VectorXd& state) const {
    if (state.size() != policy_network_->W1.cols()) {
        spdlog::error("Invalid state dimension. Expected: {}, Got: {}",
                     policy_network_->W1.cols(), state.size());
        throw std::invalid_argument("Invalid state dimension");
    }

    // Check for NaN or Inf values
    for (int i = 0; i < state.size(); ++i) {
        if (std::isnan(state(i)) || std::isinf(state(i))) {
            spdlog::error("Invalid state value at index {}: {}", i, state(i));
            throw std::invalid_argument("State contains NaN or Inf values");
        }
    }
}

// In RLAgent.cpp
TradingAction RLAgent::selectAction(const Eigen::VectorXd& state) {
    try {
        validateState(state);

        std::uniform_real_distribution<> dis(0.0, 1.0);
        if (dis(rng_) < epsilon_) {
            // Random action with different probabilities
            double rand_val = dis(rng_);
            if (rand_val < 0.4) return TradingAction::HOLD;
            else if (rand_val < 0.7) return TradingAction::BUY;
            else return TradingAction::SELL;
        }

        // Get action probabilities from policy network
        Eigen::VectorXd action_probs = policy_network_->forward(state);

        // Add small random noise to break ties
        std::normal_distribution<> noise(0, 0.01);
        for (int i = 0; i < action_probs.size(); i++) {
            action_probs(i) += noise(rng_);
        }

        // Select action with highest probability
        Eigen::Index max_idx;
        action_probs.maxCoeff(&max_idx);
        return static_cast<TradingAction>(max_idx - 1);
    } catch (const std::exception& e) {
        spdlog::error("Error in selectAction: {}", e.what());
        return TradingAction::HOLD;  // Safe default
    }
}

std::vector<double> RLAgent::computeTDError(const std::vector<Experience>& batch) {
    std::vector<double> td_errors;
    td_errors.reserve(batch.size());

    for (const auto& experience : batch) {
        Eigen::VectorXd current_q = policy_network_->forward(experience.state);
        double current_value = current_q[static_cast<int>(experience.action) + 1];

        double target_value;
        if (experience.done) {
            target_value = experience.reward;
        } else {
            Eigen::VectorXd next_q = target_network_->forward(experience.nextState);
            target_value = experience.reward + gamma_ * next_q.maxCoeff();
        }

        td_errors.push_back(target_value - current_value);
    }

    return td_errors;
}

void RLAgent::clipGradients(std::vector<Eigen::MatrixXd>& weight_gradients,
                           std::vector<Eigen::VectorXd>& bias_gradients) {
    double total_norm = 0.0;

    for (const auto& grad : weight_gradients) {
        total_norm += grad.squaredNorm();
    }
    for (const auto& grad : bias_gradients) {
        total_norm += grad.squaredNorm();
    }
    total_norm = std::sqrt(total_norm);

    if (total_norm > max_grad_norm_) {
        double scale = max_grad_norm_ / total_norm;
        for (auto& grad : weight_gradients) {
            grad *= scale;
        }
        for (auto& grad : bias_gradients) {
            grad *= scale;
        }
    }
}

void RLAgent::updateLearningRate() {
    learning_rate_ = std::max(
        min_learning_rate_,
        initial_learning_rate_ * std::pow(learning_rate_decay_, training_steps_)
    );
}

void RLAgent::updateEpsilon() {
    epsilon_ = std::max(
        min_epsilon_,
        epsilon_ * epsilon_decay_
    );
}

void RLAgent::train(size_t batch_size) {
    try {
        if (replay_buffer_->size() < batch_size) {
            return;
        }

        std::vector<Experience> batch = replay_buffer_->sample(batch_size);
        std::vector<double> td_errors = computeTDError(batch);

        std::vector<Eigen::MatrixXd> weight_gradients = {
            Eigen::MatrixXd::Zero(policy_network_->W1.rows(), policy_network_->W1.cols()),
            Eigen::MatrixXd::Zero(policy_network_->W2.rows(), policy_network_->W2.cols()),
            Eigen::MatrixXd::Zero(policy_network_->W3.rows(), policy_network_->W3.cols())
        };

        std::vector<Eigen::VectorXd> bias_gradients = {
            Eigen::VectorXd::Zero(policy_network_->b1.size()),
            Eigen::VectorXd::Zero(policy_network_->b2.size()),
            Eigen::VectorXd::Zero(policy_network_->b3.size())
        };

        double batch_loss = 0.0;

        for (size_t i = 0; i < batch.size(); ++i) {
            const auto& experience = batch[i];
            double td_error = td_errors[i];

            // Forward pass
            Eigen::VectorXd state = experience.state;
            Eigen::VectorXd h1 = relu((policy_network_->W1 * state) + policy_network_->b1);
            Eigen::VectorXd h2 = relu((policy_network_->W2 * h1) + policy_network_->b2);
            Eigen::VectorXd output = policy_network_->W3 * h2 + policy_network_->b3;
            Eigen::VectorXd probabilities = softmax(output);

            // Compute loss
            batch_loss += td_error * td_error;

            // Backpropagation
            Eigen::VectorXd d_output = probabilities;
            d_output[static_cast<int>(experience.action) + 1] -= 1.0;
            d_output *= td_error;

            Eigen::VectorXd d_h2 = (policy_network_->W3.transpose() * d_output).array()
                                  * (h2.array() > 0.0).cast<double>();
            Eigen::VectorXd d_h1 = (policy_network_->W2.transpose() * d_h2).array()
                                  * (h1.array() > 0.0).cast<double>();

            // Accumulate gradients
            weight_gradients[0] += d_h1 * state.transpose();
            weight_gradients[1] += d_h2 * h1.transpose();
            weight_gradients[2] += d_output * h2.transpose();

            bias_gradients[0] += d_h1;
            bias_gradients[1] += d_h2;
            bias_gradients[2] += d_output;
        }

        // Average gradients over batch
        double scale = 1.0 / batch_size;
        for (auto& grad : weight_gradients) grad *= scale;
        for (auto& grad : bias_gradients) grad *= scale;

        // Clip gradients
        clipGradients(weight_gradients, bias_gradients);

        // Update policy network
        policy_network_->update(weight_gradients, bias_gradients, learning_rate_);

        // Update learning rate and epsilon
        updateLearningRate();
        updateEpsilon();

        // Update metrics
        metrics_.training_losses.push_back(batch_loss / batch_size);
        for (const auto& experience : batch) {
            updateMetrics(experience);
        }

        // Update target network
        updateTargetNetwork();

        // Increment training steps
        training_steps_++;

        // Log metrics periodically
        if (training_steps_ % 1000 == 0) {
            logMetrics();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in train: " << e.what() << std::endl;
    }
}

void RLAgent::update(const Experience& experience) {
    try {
        // Add experience to replay buffer
        replay_buffer_->addExperience(experience);

        // Update metrics
        updateMetrics(experience);

        // If we have enough experiences, perform a training step
        if (replay_buffer_->size() >= 64) {  // Using batch size of 64
            train(64);
        }

        // Update epsilon (exploration rate)
        updateEpsilon();

        // Update learning rate
        updateLearningRate();

        // Increment training steps
        training_steps_++;

        // Log metrics periodically
        if (training_steps_ % 1000 == 0) {
            logMetrics();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in RLAgent::update: " << e.what() << std::endl;
    }
}

void RLAgent::updateMetrics(const Experience& experience) {
    cumulative_reward_ += experience.reward;
    metrics_.portfolio_values.push_back(experience.portfolioValue);

    if (experience.done) {
        metrics_.episode_returns.push_back(cumulative_reward_);
        cumulative_reward_ = 0.0;

        // Calculate Sharpe ratio
        if (metrics_.portfolio_values.size() > 1) {
            double avg_return = 0.0;
            double std_dev = 0.0;
            std::vector<double> returns;

            for (size_t i = 1; i < metrics_.portfolio_values.size(); ++i) {
                double daily_return = (metrics_.portfolio_values[i] - metrics_.portfolio_values[i-1])
                                    / metrics_.portfolio_values[i-1];
                returns.push_back(daily_return);
                avg_return += daily_return;
            }

            avg_return /= returns.size();

            for (double ret : returns) {
                std_dev += std::pow(ret - avg_return, 2);
            }
            std_dev = std::sqrt(std_dev / returns.size());

            double sharpe = std_dev == 0 ? 0 : (avg_return * std::sqrt(252)) / (std_dev * std::sqrt(252));
            metrics_.sharpe_ratios.push_back(sharpe);
        }

        // Calculate maximum drawdown
        double max_value = metrics_.portfolio_values[0];
        double max_drawdown = 0.0;

        for (double value : metrics_.portfolio_values) {
            if (value > max_value) {
                max_value = value;
            }
            double drawdown = (max_value - value) / max_value;
            max_drawdown = std::max(max_drawdown, drawdown);
        }

        metrics_.max_drawdowns.push_back(max_drawdown);
        metrics_.portfolio_values.clear();
    }
}

void RLAgent::updateTargetNetwork() {
    target_network_->W1 = tau_ * policy_network_->W1 + (1 - tau_) * target_network_->W1;
    target_network_->W2 = tau_ * policy_network_->W2 + (1 - tau_) * target_network_->W2;
    target_network_->W3 = tau_ * policy_network_->W3 + (1 - tau_) * target_network_->W3;

    target_network_->b1 = tau_ * policy_network_->b1 + (1 - tau_) * target_network_->b1;
    target_network_->b2 = tau_ * policy_network_->b2 + (1 - tau_) * target_network_->b2;
    target_network_->b3 = tau_ * policy_network_->b3 + (1 - tau_) * target_network_->b3;
}

void RLAgent::logMetrics() const {
    if (metrics_.episode_returns.empty()) return;

    size_t window_size = std::min(size_t(100), metrics_.episode_returns.size());

    double avg_return = getAverageReturn(window_size);
    double avg_sharpe = getAverageSharpeRatio(window_size);
    double avg_drawdown = getAverageDrawdown(window_size);

    std::cout << "\nTraining Metrics (Last " << window_size << " Episodes):\n"
              << "Average Return: " << avg_return << "\n"
              << "Average Sharpe Ratio: " << avg_sharpe << "\n"
              << "Average Max Drawdown: " << avg_drawdown << "\n"
              << "Training Steps: " << training_steps_ << "\n"
              << "Current Learning Rate: " << learning_rate_ << "\n"
              << "Current Epsilon: " << epsilon_ << std::endl;
}

double RLAgent::getAverageReturn(size_t window_size) const {
    if (metrics_.episode_returns.empty()) return 0.0;

    size_t start = std::max(size_t(0), metrics_.episode_returns.size() - window_size);
    return std::accumulate(metrics_.episode_returns.begin() + start,
                          metrics_.episode_returns.end(), 0.0) /
           (metrics_.episode_returns.end() - (metrics_.episode_returns.begin() + start));
}

double RLAgent::getAverageSharpeRatio(size_t window_size) const {
    if (metrics_.sharpe_ratios.empty()) return 0.0;

    size_t start = std::max(size_t(0), metrics_.sharpe_ratios.size() - window_size);
    return std::accumulate(metrics_.sharpe_ratios.begin() + start,
                          metrics_.sharpe_ratios.end(), 0.0) /
           (metrics_.sharpe_ratios.end() - (metrics_.sharpe_ratios.begin() + start));
}

double RLAgent::getAverageDrawdown(size_t window_size) const {
    if (metrics_.max_drawdowns.empty()) return 0.0;

    size_t start = std::max(size_t(0), metrics_.max_drawdowns.size() - window_size);
    return std::accumulate(metrics_.max_drawdowns.begin() + start,
                          metrics_.max_drawdowns.end(), 0.0) /
           (metrics_.max_drawdowns.end() - (metrics_.max_drawdowns.begin() + start));
}

bool RLAgent::shouldEarlyStop(size_t patience, double min_improvement) const {
    if (metrics_.episode_returns.size() < patience * 2) return false;

    size_t start = metrics_.episode_returns.size() - patience * 2;
    double prev_avg = std::accumulate(metrics_.episode_returns.begin() + start,
                                    metrics_.episode_returns.begin() + start + patience,
                                    0.0) / patience;
    double curr_avg = std::accumulate(metrics_.episode_returns.begin() + start + patience,
                                    metrics_.episode_returns.end(),
                                    0.0) / patience;

    return (curr_avg - prev_avg) < min_improvement;
}

void RLAgent::saveModel(const std::string& path) const {
    try {
        std::ofstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Could not open file for writing: " + path);
        }

        // Save network dimensions
        size_t input_dim = policy_network_->W1.cols();
        size_t hidden_dim = policy_network_->W1.rows();
        size_t output_dim = policy_network_->W3.rows();

        file.write(reinterpret_cast<const char*>(&input_dim), sizeof(input_dim));
        file.write(reinterpret_cast<const char*>(&hidden_dim), sizeof(hidden_dim));
        file.write(reinterpret_cast<const char*>(&output_dim), sizeof(output_dim));

        // Save hyperparameters
        file.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(learning_rate_));
        file.write(reinterpret_cast<const char*>(&gamma_), sizeof(gamma_));
        file.write(reinterpret_cast<const char*>(&epsilon_), sizeof(epsilon_));
        file.write(reinterpret_cast<const char*>(&tau_), sizeof(tau_));

        // Helper lambdas to save Eigen matrices and vectors
        auto saveMatrix = [&file](const Eigen::MatrixXd& matrix) {
            for (int i = 0; i < matrix.size(); ++i) {
                double val = *(matrix.data() + i);
                file.write(reinterpret_cast<const char*>(&val), sizeof(double));
            }
        };

        auto saveVector = [&file](const Eigen::VectorXd& vector) {
            for (int i = 0; i < vector.size(); ++i) {
                double val = vector(i);
                file.write(reinterpret_cast<const char*>(&val), sizeof(double));
            }
        };

        // Save network weights and biases
        saveMatrix(policy_network_->W1);
        saveMatrix(policy_network_->W2);
        saveMatrix(policy_network_->W3);
        saveVector(policy_network_->b1);
        saveVector(policy_network_->b2);
        saveVector(policy_network_->b3);

    } catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
        throw;
    }
}

void RLAgent::loadModel(const std::string& path) {
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Could not open file for reading: " + path);
        }

        // Load network dimensions
        size_t input_dim, hidden_dim, output_dim;
        file.read(reinterpret_cast<char*>(&input_dim), sizeof(input_dim));
        file.read(reinterpret_cast<char*>(&hidden_dim), sizeof(hidden_dim));
        file.read(reinterpret_cast<char*>(&output_dim), sizeof(output_dim));

        // Load hyperparameters
        file.read(reinterpret_cast<char*>(&learning_rate_), sizeof(learning_rate_));
        file.read(reinterpret_cast<char*>(&gamma_), sizeof(gamma_));
        file.read(reinterpret_cast<char*>(&epsilon_), sizeof(epsilon_));
        file.read(reinterpret_cast<char*>(&tau_), sizeof(tau_));

        // Helper lambdas to load Eigen matrices and vectors
        auto loadMatrix = [&file](Eigen::MatrixXd& matrix) {
            for (int i = 0; i < matrix.size(); ++i) {
                double val;
                file.read(reinterpret_cast<char*>(&val), sizeof(double));
                *(matrix.data() + i) = val;
            }
        };

        auto loadVector = [&file](Eigen::VectorXd& vector) {
            for (int i = 0; i < vector.size(); ++i) {
                double val;
                file.read(reinterpret_cast<char*>(&val), sizeof(double));
                vector(i) = val;
            }
        };

        // Create new networks with loaded dimensions
        policy_network_ = std::make_unique<Network>(input_dim, hidden_dim, output_dim);
        target_network_ = std::make_unique<Network>(input_dim, hidden_dim, output_dim);

        // Load network weights and biases
        loadMatrix(policy_network_->W1);
        loadMatrix(policy_network_->W2);
        loadMatrix(policy_network_->W3);
        loadVector(policy_network_->b1);
        loadVector(policy_network_->b2);
        loadVector(policy_network_->b3);

        // Copy policy network to target network
        updateTargetNetwork();

    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        throw;
    }
}