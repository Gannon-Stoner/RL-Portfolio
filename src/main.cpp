#include <iostream>
#include <vector>
#include <memory>
#include <filesystem>
#include "agent/RLAgent.h"
#include "environment/PortfolioEnv.h"
#include "data/DataFetcher.h"
#include "data/DataLoader.h"
#include "backtester/Backtester.h"
#include "utils/Logger.h"
#include "utils/Config.h"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

// Configuration constants
constexpr size_t BATCH_SIZE = 64;
constexpr size_t EPISODES = 1000;
constexpr double INITIAL_CAPITAL = 100000.0;
constexpr double TRANSACTION_COST = 0.001;  // 0.1% transaction cost
const std::vector<std::string> SYMBOLS = {"AAPL", "MSFT", "GOOGL", "AMZN", "META"};

class PortfolioOptimizer {
public:
    PortfolioOptimizer(const std::string& api_key, const Config::TradingParams& params)
        : logger_(std::make_shared<Logger>("logs/portfolio_optimization.log"))
        , data_fetcher_(std::make_shared<DataFetcher>(api_key))
        , data_loader_(std::make_shared<DataLoader>())
        , trading_params_(params) {

        setupDirectories();
        setupComponents();
    }

    void run() {
        try {
            // 1. Fetch and prepare data
            if (!loadOrFetchData()) {
                logger_->error("Failed to load or fetch data");
                return;
            }

            // 2. Train the agent
            trainAgent();

            // 3. Evaluate performance
            evaluatePerformance();

        } catch (const std::exception& e) {
            logger_->error("Error in portfolio optimization: " + std::string(e.what()));
        }
    }

private:
    void setupDirectories() {
        std::vector<std::string> dirs = {
            "data",
            "models",
            "logs",
            "reports"
        };

        for (const auto& dir : dirs) {
            std::filesystem::create_directories(dir);
        }
    }

    bool loadOrFetchData() {
    std::vector<std::string> data_files;
    bool needs_fetch = false;

    // Check if we need to fetch new data
    for (const auto& symbol : trading_params_.symbols) {
        std::string filename = "data/" + symbol + "_daily.csv";
        data_files.push_back(filename);

        if (!std::filesystem::exists(filename)) {
            needs_fetch = true;
            break;
        }
    }

    if (needs_fetch) {
        logger_->info("Fetching new market data...");
        for (size_t i = 0; i < trading_params_.symbols.size(); ++i) {
            std::vector<std::vector<std::string>> symbol_data;
            if (!data_fetcher_->fetchDailyData(trading_params_.symbols[i], "full", symbol_data)) {
                logger_->error_fmt("Failed to fetch data for symbol: {}", trading_params_.symbols[i]);
                return false;
            }

            // Save to CSV
            std::ofstream file(data_files[i]);
            if (!file.is_open()) {
                logger_->error_fmt("Failed to create file: {}", data_files[i]);
                return false;
            }

            file << "timestamp,open,high,low,close,volume\n";
            for (const auto& row : symbol_data) {
                for (size_t j = 0; j < row.size(); ++j) {
                    file << row[j] << (j < row.size() - 1 ? "," : "\n");
                }
            }
        }
        logger_->info("Market data saved to files");
    }

    // Load data using DataLoader
    data_loader_->loadMultipleAssets(data_files);

    // Verify data was loaded
    if (data_loader_->getPrices().empty()) {
        logger_->error("Failed to load price data");
        return false;
    }

    logger_->info_fmt("Successfully loaded data for {} assets", trading_params_.symbols.size());
    return true;
}

    void setupComponents() {
        // Make sure data is loaded before setting up components
        if (!loadOrFetchData()) {
            throw std::runtime_error("Failed to load market data");
        }

        // Initialize replay buffer
        replay_buffer_ = std::make_shared<ReplayBuffer>(10000);

        // Calculate dimensions based on loaded data
        size_t num_assets = trading_params_.symbols.size();

        // State dimension breakdown:
        size_t price_dim = num_assets;
        size_t holdings_dim = num_assets;
        size_t tech_dim = num_assets * NUM_TECHNICAL_FEATURES;
        size_t cash_dim = 1;

        size_t total_state_dim = price_dim + holdings_dim + tech_dim + cash_dim;

        logger_->info("State dimensions breakdown:");
        logger_->info_fmt("Number of assets: {}", num_assets);
        logger_->info_fmt("Price dimensions: {}", price_dim);
        logger_->info_fmt("Holdings dimensions: {}", holdings_dim);
        logger_->info_fmt("Technical feature dimensions: {}", tech_dim);
        logger_->info_fmt("Cash dimension: {}", cash_dim);
        logger_->info_fmt("Total state dimensions: {}", total_state_dim);

        // Initialize agent
        agent_ = std::make_shared<RLAgent>(
            total_state_dim,
            num_assets,  // action dimension is number of assets
            replay_buffer_,
            trading_params_.learning_rate,
            trading_params_.gamma,
            trading_params_.epsilon,
            0.01
        );

        // Initialize environment with loaded data
        env_ = std::make_shared<PortfolioEnv>(
            data_loader_,
            trading_params_.initial_capital,
            trading_params_.transaction_cost,
            trading_params_.max_position_size
        );

        // Initialize backtester
        backtester_ = std::make_shared<Backtester>(agent_, env_, logger_);
    }

    void trainAgent() {
        logger_->info("Starting agent training...");

        for (size_t episode = 0; episode < EPISODES; ++episode) {
            auto env_state = env_->reset();
            // Convert environment state to vector for the agent
            Eigen::VectorXd state = backtester_->stateToVector(env_state);
            double episode_reward = 0.0;
            bool done = false;

            while (!done) {
                // Select action using the vectorized state
                auto action = agent_->selectAction(state);

                // Execute action in environment
                auto [next_env_state, reward, is_done] = env_->step(backtester_->actionToVector(action));

                // Convert next state to vector for the agent
                Eigen::VectorXd next_state = backtester_->stateToVector(next_env_state);

                // Store experience using vectorized states
                Experience exp(state, action, reward, next_state, is_done);
                exp.portfolioValue = env_->getPortfolioValue();
                replay_buffer_->addExperience(exp);

                // Train agent if we have enough experiences
                if (replay_buffer_->size() >= BATCH_SIZE) {
                    agent_->train(BATCH_SIZE);
                }

                state = next_state;
                episode_reward += reward;
                done = is_done;
            }

            // Log episode results
            std::string metrics = "Episode: " + std::to_string(episode) +
                                ", Return: " + std::to_string(episode_reward) +
                                ", Portfolio Value: " + std::to_string(env_->getPortfolioValue());
            logger_->logTrainingMetrics(metrics);

            // Save model periodically
            if (episode % 100 == 0) {
                std::string model_path = "models/agent_" + std::to_string(episode) + ".model";
                agent_->saveModel(model_path);
                logger_->info("Saved model to: " + model_path);
            }
        }
    }

    void evaluatePerformance() {
        logger_->info("Evaluating agent performance...");

        // Get the last year of data for testing
        auto dates = data_loader_->getAlignedDates();
        std::string test_start = dates[dates.size() * 4 / 5];  // Last 20% of data
        std::string test_end = dates.back();

        // Run backtest
        auto result = backtester_->runBacktest(test_start, test_end, false);

        // Generate and save performance report
        backtester_->generatePerformanceReport(result, "reports/performance_report.txt");
        backtester_->plotPerformanceMetrics(result, "reports/performance_plots");

        // Log final metrics
        std::string final_metrics =
            "Total Return: " + std::to_string(result.total_return) + "\n" +
            "Sharpe Ratio: " + std::to_string(result.sharpe_ratio) + "\n" +
            "Max Drawdown: " + std::to_string(result.max_drawdown) + "\n" +
            "Volatility: " + std::to_string(result.volatility);

        logger_->logBacktestResults(final_metrics);
    }

    Config::TradingParams trading_params_;
    std::shared_ptr<Logger> logger_;
    std::shared_ptr<DataFetcher> data_fetcher_;
    std::shared_ptr<DataLoader> data_loader_;
    std::shared_ptr<ReplayBuffer> replay_buffer_;
    std::shared_ptr<RLAgent> agent_;
    std::shared_ptr<PortfolioEnv> env_;
    std::shared_ptr<Backtester> backtester_;
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <alpha_vantage_api_key>\n";
        return 1;
    }

    try {
        // Load config from the default location
        auto config = Config::loadConfig("config.json");
        PortfolioOptimizer optimizer(argv[1], config);
        optimizer.run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}