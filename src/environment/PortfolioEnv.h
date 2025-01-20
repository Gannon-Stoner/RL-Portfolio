//
// Created by grs on 1/14/25.
//

#ifndef PORTFOLIOENV_H
#define PORTFOLIOENV_H

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <tuple>
#include <spdlog/spdlog.h>
#include "../data/DataLoader.h"
#include <sstream>

// Environment configuration constants
constexpr size_t LOOKBACK_WINDOW = 20;  // For calculating features
constexpr int NUM_TECHNICAL_FEATURES = 3;  // SMA, Volatility, Momentum

class PortfolioEnv {
public:
    struct State {
        Eigen::VectorXd prices;     // Current asset prices
        Eigen::VectorXd holdings;   // Current portfolio holdings
        Eigen::MatrixXd features;   // Technical indicators, market features
        double cash;                // Available cash

        // Historical data for feature calculation
        std::vector<Eigen::VectorXd> price_history;   // Last LOOKBACK_WINDOW prices
        std::vector<Eigen::VectorXd> volume_history;  // Last LOOKBACK_WINDOW volumes

        State(size_t num_assets, size_t num_features)
            : prices(Eigen::VectorXd::Zero(num_assets))
            , holdings(Eigen::VectorXd::Zero(num_assets))
            , features(Eigen::MatrixXd::Zero(num_assets, num_features))
            , cash(0.0) {}

        // Helper methods
        double getPortfolioValue() const {
            return cash + (prices.array() * holdings.array()).sum();
        }

        Eigen::VectorXd getCurrentWeights() const {
            double total_value = getPortfolioValue();
            if (total_value == 0.0) return Eigen::VectorXd::Zero(holdings.size());
            return (prices.array() * holdings.array()) / total_value;
        }
    };

    PortfolioEnv(std::shared_ptr<DataLoader> data_loader,
                 double initial_capital,
                 double transaction_cost_rate = 0.001,
                 double max_position_size = 0.5)
        : current_state_(0, NUM_TECHNICAL_FEATURES)  // Will resize after loading data
        , current_step_(0)
        , transaction_cost_rate_(transaction_cost_rate)
        , max_position_size_(max_position_size)
    {
        // Load market data from the DataLoader
        historical_prices_ = data_loader->getPrices();
        historical_volumes_ = data_loader->getVolumes();

        if (historical_prices_.empty()) {
            throw std::runtime_error("No price data loaded");
        }

        // Resize state vectors based on loaded data
        size_t num_assets = historical_prices_[0].size();
        current_state_ = State(num_assets, NUM_TECHNICAL_FEATURES);
        current_state_.cash = initial_capital;
        current_state_.holdings = Eigen::VectorXd::Zero(num_assets);
        current_state_.prices = historical_prices_[0];

        spdlog::info("PortfolioEnv initialized with:");
        spdlog::info("  Number of assets: {}", num_assets);
        spdlog::info("  Number of time steps: {}", historical_prices_.size());
        spdlog::info("  Initial capital: {}", initial_capital);
    }

    PortfolioEnv(const std::string& data_path,
                 double initial_capital,
                 double transaction_cost_rate = 0.001,
                 double max_position_size = 0.5);

    // Core RL environment methods
    State reset();
    std::tuple<State, double, bool> step(const Eigen::VectorXd& action);

    // Utility methods
    double getPortfolioValue() const { return current_state_.getPortfolioValue(); }
    Eigen::VectorXd getCurrentWeights() const { return current_state_.getCurrentWeights(); }
    size_t getNumAssets() const { return historical_prices_[0].size(); }

private:
    // Internal methods
    void loadMarketData(const std::string& data_path);
    void executeRebalancing(const Eigen::VectorXd& target_weights);

    // Feature calculation methods
    Eigen::MatrixXd calculateFeatures();
    double calculateSMA(int asset_idx, int window, bool use_volume = false);
    double calculateVolatility(int asset_idx, int window);
    double calculateMomentum(int asset_idx, int window);

    // Risk-adjusted reward calculation
    double calculateReward(double old_value, double new_value) const;
    double calculateSharpeRatio() const;

    // Validation methods
    bool validateAction(const Eigen::VectorXd& action) const;

    // Member variables
    State current_state_;
    std::vector<Eigen::VectorXd> historical_prices_;
    std::vector<Eigen::VectorXd> historical_volumes_;
    size_t current_step_;
    double transaction_cost_rate_;
    double max_position_size_;  // Maximum allowed position size as fraction of portfolio

    // Risk management parameters
    std::vector<double> portfolio_values_;  // Track historical portfolio values
    double risk_free_rate_ = 0.02;  // Annual risk-free rate for Sharpe ratio

    // Helper function to convert Eigen vector to string
    static std::string vectorToString(const Eigen::VectorXd& vec) {
        std::ostringstream oss;
        oss << "[";
        for (int i = 0; i < vec.size(); ++i) {
            oss << vec(i);
            if (i < vec.size() - 1) oss << ", ";
        }
        oss << "]";
        return oss.str();
    }
};

#endif //PORTFOLIOENV_H
