//
// Created by grs on 1/14/25.
//

#include "Backtester.h"
#include <cmath>
#include <fstream>
#include <numeric>
#include <iomanip>

Backtester::Backtester(std::shared_ptr<RLAgent> agent,
                       std::shared_ptr<PortfolioEnv> env,
                       std::shared_ptr<Logger> logger)
    : agent_(agent)
    , env_(env)
    , logger_(logger) {}

// Helper method to convert PortfolioEnv::State to Eigen::VectorXd
Eigen::VectorXd Backtester::stateToVector(const PortfolioEnv::State& state) const {
    // Calculate total size needed for the vector
    size_t num_assets = state.prices.size();
    size_t total_size = num_assets +                    // Asset prices
                       num_assets +                      // Current holdings
                       (num_assets * NUM_TECHNICAL_FEATURES) +  // Technical indicators
                       1;                                // Cash position

    Eigen::VectorXd state_vector(total_size);
    size_t idx = 0;

    // Add prices
    state_vector.segment(idx, num_assets) = state.prices;
    idx += num_assets;

    // Add holdings
    state_vector.segment(idx, num_assets) = state.holdings;
    idx += num_assets;

    // Add technical features (flatten the matrix)
    for (int i = 0; i < state.features.rows(); ++i) {
        for (int j = 0; j < state.features.cols(); ++j) {
            state_vector(idx++) = state.features(i, j);
        }
    }

    // Add cash position
    state_vector(idx) = state.cash;

    // Debug logging
    spdlog::debug("State vector construction:");
    spdlog::debug("Prices segment size: {}", num_assets);
    spdlog::debug("Holdings segment size: {}", num_assets);
    spdlog::debug("Technical features segment size: {}", num_assets * NUM_TECHNICAL_FEATURES);
    spdlog::debug("Total state vector size: {}", total_size);

    return state_vector;
}

Eigen::VectorXd Backtester::actionToVector(TradingAction action) const {
    Eigen::VectorXd action_vector = Eigen::VectorXd::Zero(env_->getNumAssets());
    double position_size = env_->getMaxPositionSize();  // Should be 0.5 from config
    size_t num_assets = env_->getNumAssets();  // Should be 3

    // Log current portfolio state
    Eigen::VectorXd current_weights = env_->getCurrentWeights();
    spdlog::info("Current weights before action: [{}, {}, {}]",
                 current_weights(0), current_weights(1), current_weights(2));

    switch (action) {
        case TradingAction::SELL:
            // Keep very small positions (10% total portfolio split equally)
            action_vector.setConstant(0.1 / num_assets);
            spdlog::info("SELL action selected");
            break;

        case TradingAction::HOLD:
            // If holding, use current weights but ensure they're balanced
            if (current_weights.sum() < 0.1) {  // If mostly in cash
                action_vector.setConstant(0.3 / num_assets);  // Put 30% to work
            } else {
                action_vector = current_weights;
            }
            spdlog::info("HOLD action selected");
            break;

        case TradingAction::BUY:
            // Take larger positions but respect position size limits
            double weight_per_asset = std::min(position_size, 0.8 / num_assets);
            action_vector.setConstant(weight_per_asset);
            spdlog::info("BUY action selected");
            break;
    }

    // Ensure no single position exceeds position_size
    for (int i = 0; i < num_assets; i++) {
        if (action_vector(i) > position_size) {
            action_vector(i) = position_size;
        }
    }

    // Calculate total allocation
    double total_allocation = action_vector.sum();

    // If we're allocating more than 100%, normalize
    if (total_allocation > 1.0) {
        action_vector *= (0.99 / total_allocation);  // Leave a small cash buffer
    }

    // Log final weights
    spdlog::info("Final action weights: [{}, {}, {}], Total allocation: {}",
                 action_vector(0), action_vector(1), action_vector(2),
                 action_vector.sum());

    return action_vector;
}

Backtester::BacktestResult Backtester::runBacktest(
    const std::string& start_date,
    const std::string& end_date,
    bool training_mode) {

    BacktestResult result;
    PortfolioEnv::State current_state = env_->reset();
    bool done = false;
    double initial_value = env_->getPortfolioValue();

    while (!done) {
        // Convert environment state to vector for the agent
        Eigen::VectorXd state_vector = stateToVector(current_state);

        // Get action from agent and convert to action vector
        TradingAction agent_action = agent_->selectAction(state_vector);
        Eigen::VectorXd action_vector = actionToVector(agent_action);

        // Execute action and get new state
        auto step_result = env_->step(action_vector);
        PortfolioEnv::State next_state = std::get<0>(step_result);
        double reward = std::get<1>(step_result);
        bool is_done = std::get<2>(step_result);

        // Record results
        double portfolio_value = env_->getPortfolioValue();
        result.portfolio_values.push_back(portfolio_value);
        result.portfolio_weights.push_back(env_->getCurrentWeights());
        result.actions.push_back(agent_action);

        // Calculate return if we have at least two values
        if (result.portfolio_values.size() > 1) {
            double daily_return = (portfolio_value - result.portfolio_values[result.portfolio_values.size()-2]) /
                                 result.portfolio_values[result.portfolio_values.size()-2];
            result.returns.push_back(daily_return);
        }

        // If in training mode, store experience
        if (training_mode) {
            Eigen::VectorXd next_state_vector = stateToVector(next_state);
            Experience exp(state_vector, agent_action, reward, next_state_vector, is_done);
            exp.portfolioValue = portfolio_value;
            agent_->update(exp);
        }

        // Update state and done flag
        current_state = next_state;
        done = is_done;
    }

    // Calculate final metrics
    result.total_return = (result.portfolio_values.back() - initial_value) / initial_value;
    result.sharpe_ratio = calculateSharpeRatio(result.returns);
    result.max_drawdown = calculateMaxDrawdown(result.portfolio_values);
    result.volatility = calculateVolatility(result.returns);
    result.sortino_ratio = calculateSortinoRatio(result.returns);

    return result;
}

void Backtester::generatePerformanceReport(
    const BacktestResult& result,
    const std::string& output_path) const {

    std::ofstream report(output_path);
    if (!report) {
        logger_->error("Failed to create performance report file: " + output_path);
        return;
    }

    report << std::fixed << std::setprecision(4);
    report << "Portfolio Performance Report\n";
    report << "==========================\n\n";

    report << "Summary Statistics:\n";
    report << "-----------------\n";
    report << "Total Return: " << (result.total_return * 100) << "%\n";
    report << "Sharpe Ratio: " << result.sharpe_ratio << "\n";
    report << "Sortino Ratio: " << result.sortino_ratio << "\n";
    report << "Maximum Drawdown: " << (result.max_drawdown * 100) << "%\n";
    report << "Volatility (Annualized): " << (result.volatility * std::sqrt(252) * 100) << "%\n\n";

    report << "Portfolio Value History:\n";
    report << "---------------------\n";
    for (size_t i = 0; i < result.portfolio_values.size(); ++i) {
        report << "Day " << i << ": $" << result.portfolio_values[i] << "\n";
    }

    report << "\nTrading Actions Summary:\n";
    report << "---------------------\n";
    size_t buys = 0, sells = 0, holds = 0;
    for (const auto& action : result.actions) {
        switch (action) {
            case TradingAction::BUY: ++buys; break;
            case TradingAction::SELL: ++sells; break;
            case TradingAction::HOLD: ++holds; break;
        }
    }
    report << "Total Buys: " << buys << "\n";
    report << "Total Sells: " << sells << "\n";
    report << "Total Holds: " << holds << "\n";
}

void Backtester::plotPerformanceMetrics(
    const BacktestResult& result,
    const std::string& output_path) const {
    // Note: Implementation would depend on your plotting library choice
    // For now, we'll just save the data in a format suitable for plotting
    std::ofstream data(output_path + ".csv");
    if (!data) {
        logger_->error("Failed to create performance metrics data file: " + output_path);
        return;
    }

    data << "Day,PortfolioValue,Return\n";
    for (size_t i = 0; i < result.portfolio_values.size(); ++i) {
        data << i << ","
             << result.portfolio_values[i] << ","
             << (i > 0 ? result.returns[i-1] : 0.0) << "\n";
    }
}

double Backtester::calculateSharpeRatio(const std::vector<double>& returns) const {
    if (returns.empty()) return 0.0;

    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double squared_sum = std::accumulate(returns.begin(), returns.end(), 0.0,
        [mean_return](double acc, double ret) {
            return acc + std::pow(ret - mean_return, 2);
        });

    double std_dev = std::sqrt(squared_sum / returns.size());
    if (std_dev == 0.0) return 0.0;

    // Annualize Sharpe Ratio
    return (mean_return * std::sqrt(252)) / (std_dev) * std::sqrt(252);  // Annualize by multiplying by sqrt(252 trading days)
}

double Backtester::calculateSortinoRatio(const std::vector<double>& returns) const {
    if (returns.empty()) return 0.0;

    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();

    // Calculate downside deviation (only consider negative returns)
    double squared_downside_sum = std::accumulate(returns.begin(), returns.end(), 0.0,
        [mean_return](double acc, double ret) {
            double downside = std::min(ret - mean_return, 0.0);
            return acc + std::pow(downside, 2);
        });

    double downside_deviation = std::sqrt(squared_downside_sum / returns.size());
    if (downside_deviation == 0.0) return 0.0;

    // Annualize Sortino Ratio
    return (mean_return * std::sqrt(252)) / (downside_deviation * std::sqrt(252));
}

double Backtester::calculateMaxDrawdown(const std::vector<double>& portfolio_values) const {
    if (portfolio_values.empty()) return 0.0;

    double max_drawdown = 0.0;
    double peak = portfolio_values[0];

    for (double value : portfolio_values) {
        if (value > peak) {
            peak = value;
        } else {
            double drawdown = (peak - value) / peak;
            max_drawdown = std::max(max_drawdown, drawdown);
        }
    }

    return max_drawdown;
}

double Backtester::calculateVolatility(const std::vector<double>& returns) const {
    if (returns.empty()) return 0.0;

    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();

    double squared_sum = std::accumulate(returns.begin(), returns.end(), 0.0,
        [mean_return](double acc, double ret) {
            return acc + std::pow(ret - mean_return, 2);
        });

    // Calculate annualized volatility
    return std::sqrt(squared_sum / returns.size() * 252);  // Annualize by multiplying by sqrt(252)
}