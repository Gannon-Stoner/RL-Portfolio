//
// Created by grs on 1/14/25.
//

#ifndef BACKTESTER_H
#define BACKTESTER_H

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "../agent/RLAgent.h"
#include "../environment/PortfolioEnv.h"
#include "../utils/Logger.h"

class Backtester {
public:
    struct BacktestResult {
        double total_return;
        double sharpe_ratio;
        double max_drawdown;
        double volatility;
        double sortino_ratio;
        std::vector<double> portfolio_values;
        std::vector<Eigen::VectorXd> portfolio_weights;
        std::vector<TradingAction> actions;
        std::vector<double> returns;
    };

    Backtester(std::shared_ptr<RLAgent> agent,
               std::shared_ptr<PortfolioEnv> env,
               std::shared_ptr<Logger> logger);

    // Core backtest functionality
    BacktestResult runBacktest(const std::string& start_date,
                              const std::string& end_date,
                              bool training_mode = false);

    // Analysis methods
    void generatePerformanceReport(const BacktestResult& result,
                                 const std::string& output_path) const;
    void plotPerformanceMetrics(const BacktestResult& result,
                               const std::string& output_path) const;

    // State conversion methods (moved from private to public)
    Eigen::VectorXd stateToVector(const PortfolioEnv::State& state) const;
    Eigen::VectorXd actionToVector(TradingAction action) const;

private:
    std::shared_ptr<RLAgent> agent_;
    std::shared_ptr<PortfolioEnv> env_;
    std::shared_ptr<Logger> logger_;

    // Performance calculation methods
    double calculateSharpeRatio(const std::vector<double>& returns) const;
    double calculateSortinoRatio(const std::vector<double>& returns) const;
    double calculateMaxDrawdown(const std::vector<double>& portfolio_values) const;
    double calculateVolatility(const std::vector<double>& returns) const;
};
#endif //BACKTESTER_H