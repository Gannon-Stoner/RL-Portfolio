//
// Created by grs on 1/14/25.
//

#include "PortfolioEnv.h"
#include "../data/DataLoader.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>
#include <numeric>

PortfolioEnv::PortfolioEnv(const std::string& data_path,
                          double initial_capital,
                          double transaction_cost_rate,
                          double max_position_size)
    : current_state_(0, NUM_TECHNICAL_FEATURES)  // Will resize after loading data
    , current_step_(0)
    , transaction_cost_rate_(transaction_cost_rate)
    , max_position_size_(max_position_size)
{
    loadMarketData(data_path);

    // Resize state vectors based on loaded data
    size_t num_assets = historical_prices_[0].size();
    current_state_ = State(num_assets, NUM_TECHNICAL_FEATURES);
    current_state_.cash = initial_capital;

    spdlog::info("Initialized PortfolioEnv with {} assets", num_assets);
}

void PortfolioEnv::loadMarketData(const std::string& data_path) {
    DataLoader loader;
    loader.loadMultipleAssets({data_path});  // Assuming single file for now

    historical_prices_ = loader.getPrices();
    // Note: You'll need to add getVolumes() to your DataLoader class
    historical_volumes_ = loader.getVolumes();

    if (historical_prices_.empty()) {
        throw std::runtime_error("No price data loaded");
    }

    spdlog::info("Loaded {} timesteps of market data", historical_prices_.size());
}

PortfolioEnv::State PortfolioEnv::reset() {
    current_step_ = 0;
    portfolio_values_.clear();

    // Reset state
    current_state_.cash = current_state_.getPortfolioValue();  // Convert all to cash
    current_state_.holdings = Eigen::VectorXd::Zero(getNumAssets());
    current_state_.prices = historical_prices_[0];

    // Initialize price and volume history
    current_state_.price_history.clear();
    current_state_.volume_history.clear();

    // Fill initial history buffer with first LOOKBACK_WINDOW days of data
    size_t initial_history = std::min(LOOKBACK_WINDOW, historical_prices_.size());
    for (size_t i = 0; i < initial_history; ++i) {
        if (i < historical_prices_.size()) {  // Safety check
            current_state_.price_history.push_back(historical_prices_[i]);
        }
        if (i < historical_volumes_.size()) {  // Safety check
            current_state_.volume_history.push_back(historical_volumes_[i]);
        }
    }

    // Calculate initial features
    current_state_.features = calculateFeatures();

    // Record initial portfolio value
    portfolio_values_.push_back(current_state_.getPortfolioValue());

    return current_state_;
}

// In PortfolioEnv.cpp
std::tuple<PortfolioEnv::State, double, bool> PortfolioEnv::step(const Eigen::VectorXd& action) {
    if (!validateAction(action)) {
        throw std::runtime_error("Invalid action provided");
    }

    // Store current portfolio value for reward calculation
    double old_value = current_state_.getPortfolioValue();

    // Execute trades based on target weights
    executeRebalancing(action);

    // Move to next timestep
    current_step_++;
    bool done = current_step_ >= historical_prices_.size();

    if (!done) {
        // Update state with new prices and features
        updateState();
    }

    // Calculate reward
    double new_value = current_state_.getPortfolioValue();
    double reward = calculateReward(old_value, new_value);

    return {current_state_, reward, done};
}

void PortfolioEnv::executeRebalancing(const Eigen::VectorXd& target_weights) {
    double portfolio_value = current_state_.getPortfolioValue();
    Eigen::VectorXd current_weights = current_state_.getCurrentWeights();

    // Calculate required trades
    Eigen::VectorXd trade_weights = target_weights - current_weights;

    // Calculate trading costs
    double total_trade_value = (trade_weights.array().abs() * portfolio_value).sum();
    double trading_cost = total_trade_value * transaction_cost_rate_;

    // Adjust portfolio value for trading costs
    portfolio_value -= trading_cost;
    current_state_.cash -= trading_cost;

    // Update holdings based on new weights
    for (int i = 0; i < current_state_.holdings.size(); i++) {
        if (current_state_.prices(i) > 0) {  // Prevent division by zero
            current_state_.holdings(i) = (target_weights(i) * portfolio_value) /
                                       current_state_.prices(i);
        }
    }

    // Update cash position
    double total_holdings_value = (current_state_.prices.array() *
                                 current_state_.holdings.array()).sum();
    current_state_.cash = portfolio_value - total_holdings_value;
}

bool PortfolioEnv::validateAction(const Eigen::VectorXd& action) const {
    // Check dimensions
    if (action.size() != getNumAssets()) {
        spdlog::warn("Invalid action dimension: {} (expected {})",
                    action.size(), getNumAssets());
        return false;
    }

    // Check if weights sum to 1 (within numerical precision)
    double sum = action.sum();
    if (std::abs(sum - 1.0) > 1e-5) {
        spdlog::warn("Action weights do not sum to 1: {} (weights: {})",
                    sum, vectorToString(action));
        return false;
    }

    // Check position size limits
    if ((action.array() > max_position_size_).any()) {
        spdlog::warn("Action contains position larger than maximum allowed: {}",
                    vectorToString(action));
        return false;
    }

    // Check for negative weights (no short selling)
    if ((action.array() < 0).any()) {
        spdlog::warn("Action contains negative weights: {}",
                    vectorToString(action));
        return false;
    }

    return true;
}

// Add to PortfolioEnv.cpp

Eigen::MatrixXd PortfolioEnv::calculateFeatures() {
    size_t num_assets = historical_prices_[0].size();
    Eigen::MatrixXd features(num_assets, NUM_TECHNICAL_FEATURES);

    // Calculate features for each asset
    for (size_t i = 0; i < num_assets; ++i) {
        // Feature 1: 20-day Simple Moving Average (normalized)
        features(i, 0) = calculateSMA(i, 20);

        // Feature 2: 20-day Volatility
        features(i, 1) = calculateVolatility(i, 20);

        // Feature 3: 10-day Momentum
        features(i, 2) = calculateMomentum(i, 10);
    }

    return features;
}

double PortfolioEnv::calculateSMA(int asset_idx, int window, bool use_volume) {
    if (current_state_.price_history.size() < window) {
        return 0.0;
    }

    double sum = 0.0;
    size_t start_idx = current_state_.price_history.size() - window;

    for (size_t i = start_idx; i < current_state_.price_history.size(); ++i) {
        if (use_volume) {
            sum += current_state_.volume_history[i](asset_idx);
        } else {
            sum += current_state_.price_history[i](asset_idx);
        }
    }

    double sma = sum / window;

    // Normalize by current price/volume
    double current_value = use_volume ?
        current_state_.volume_history.back()(asset_idx) :
        current_state_.price_history.back()(asset_idx);

    return current_value == 0 ? 0 : (sma / current_value - 1.0);
}

double PortfolioEnv::calculateVolatility(int asset_idx, int window) {
    if (current_state_.price_history.size() < window + 1) {
        return 0.0;
    }

    std::vector<double> returns;
    returns.reserve(window);

    // Calculate daily returns
    for (size_t i = current_state_.price_history.size() - window;
         i < current_state_.price_history.size(); ++i) {
        double today = current_state_.price_history[i](asset_idx);
        double yesterday = current_state_.price_history[i-1](asset_idx);

        if (yesterday != 0) {
            returns.push_back(today / yesterday - 1.0);
        }
    }

    // Calculate standard deviation
    if (returns.empty()) return 0.0;

    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double sq_sum = std::inner_product(returns.begin(), returns.end(),
                                     returns.begin(), 0.0);
    double variance = (sq_sum / returns.size()) - (mean * mean);

    return std::sqrt(variance * 252.0); // Annualize volatility
}

double PortfolioEnv::calculateMomentum(int asset_idx, int window) {
    if (current_state_.price_history.size() < window) {
        return 0.0;
    }

    size_t last_idx = current_state_.price_history.size() - 1;
    double current_price = current_state_.price_history[last_idx](asset_idx);
    double past_price = current_state_.price_history[last_idx - window + 1](asset_idx);

    if (past_price == 0) return 0.0;

    // Return momentum as percentage change
    return (current_price / past_price - 1.0);
}

// In PortfolioEnv.cpp
double PortfolioEnv::calculateReward(double old_value, double new_value) const {
    // Calculate return
    double returns = new_value / old_value - 1.0;

    // Calculate Sharpe ratio if we have enough history
    double sharpe = 0.0;
    if (portfolio_values_.size() >= 30) {
        sharpe = calculateSharpeRatio();
    }

    // Calculate position diversity bonus
    Eigen::VectorXd weights = current_state_.getCurrentWeights();
    double diversity = 1.0 - (weights.array() * weights.array()).sum();  // Higher for diverse portfolios

    // Combine components
    double reward = returns * (1.0 + std::max(0.0, sharpe)) + 0.1 * diversity;

    return reward;
}

double PortfolioEnv::calculateSharpeRatio() const {
    if (portfolio_values_.size() < 2) return 0.0;

    std::vector<double> returns;
    returns.reserve(portfolio_values_.size() - 1);

    // Calculate daily returns
    for (size_t i = 1; i < portfolio_values_.size(); ++i) {
        returns.push_back(portfolio_values_[i] / portfolio_values_[i-1] - 1.0);
    }

    // Calculate mean return
    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) /
                        returns.size();

    // Calculate standard deviation
    double sq_sum = std::inner_product(returns.begin(), returns.end(),
                                     returns.begin(), 0.0);
    double std_dev = std::sqrt(sq_sum/returns.size() - mean_return*mean_return);

    // Annualize Sharpe ratio, we use 252 because there are 252 trading days in a year
    double excess_return = mean_return - risk_free_rate_ / 252.0;  // Daily risk-free rate
    return std_dev == 0 ? 0 : (excess_return / std_dev) * std::sqrt(252.0);
}