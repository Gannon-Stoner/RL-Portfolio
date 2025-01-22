# RL Portfolio Optimization

A robust C++ implementation of a Deep Reinforcement Learning framework for portfolio optimization and automated trading. This project uses Deep Q-Networks (DQN) to learn optimal trading strategies across multiple assets while considering transaction costs and risk management.

## Features

- **Deep Q-Network (DQN) Implementation**
  - Experience replay for stable learning
  - Target network to reduce overestimation
  - Epsilon-greedy exploration strategy
  - Risk-adjusted rewards using Sharpe ratio

- **Market Data Integration**
  - Alpha Vantage API integration for real market data
  - Support for multiple assets (stocks, ETFs)
  - Technical indicator calculations (SMA, Volatility, Momentum)

- **Portfolio Management**
  - Dynamic position sizing
  - Transaction cost consideration
  - Risk management through position limits
  - Multi-asset portfolio balancing

- **Performance Analysis**
  - Sharpe ratio calculation
  - Maximum drawdown tracking
  - Portfolio value monitoring
  - Trading action analysis

## Dependencies

- Eigen3 (Linear algebra)
- spdlog (Logging)
- CURL (Data fetching)
- nlohmann_json (JSON parsing)
- CMake >= 3.30

## Building the Project

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

```bash
./RL_Portfolio <alpha_vantage_api_key>
```

The program requires a valid Alpha Vantage API key for fetching market data. You can get one for free at [Alpha Vantage](https://www.alphavantage.co/).

### Configuration

Modify `config.json` to customize:
- Initial capital
- Transaction costs
- Position size limits
- Training parameters
- Trading symbols

Example configuration:
```json
{
  "initial_capital": 100000.0,
  "transaction_cost": 0.001,
  "max_position_size": 0.5,
  "batch_size": 64,
  "learning_rate": 0.001,
  "gamma": 0.99,
  "epsilon": 0.1,
  "symbols": ["AAPL", "MSFT", "GOOGL"]
}
```

## Project Structure

```
├── src/
│   ├── agent/          # DQN agent implementation
│   ├── environment/    # Trading environment
│   ├── data/          # Data handling and processing
│   ├── backtester/    # Performance evaluation
│   ├── memory/        # Experience replay buffer
│   ├── utils/         # Logging and configuration
│   └── main.cpp       # Entry point
├── data/              # Market data storage
├── models/            # Saved model states
├── logs/             # Execution logs
└── reports/          # Performance reports
```

## Features in Detail

### DQN Agent
- Three possible actions per asset: BUY, HOLD, SELL
- State space includes:
  - Current prices
  - Portfolio holdings
  - Technical indicators
  - Cash position

### Environment
- Supports multiple assets
- Calculates transaction costs
- Enforces position limits
- Provides risk-adjusted rewards

### Performance Metrics
- Total return
- Sharpe ratio
- Maximum drawdown
- Portfolio volatility
- Position diversity

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Future Improvements

- Support for more sophisticated DRL algorithms (A2C, PPO)
- Integration with additional data sources
- Real-time trading capabilities
- Enhanced technical indicators
- Options and derivatives support
- Risk-adjusted position sizing
