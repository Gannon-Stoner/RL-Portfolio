//
// Created by grs on 1/15/25.
//

#ifndef REPLAYBUFFER_H
#define REPLAYBUFFER_H
#include <vector>
#include <random>
#include <deque>
#include <Eigen/Dense>
#include <string>
#include <algorithm>
#include <stdexcept>

enum class TradingAction {
  SELL = -1,
  HOLD = 0,
  BUY = 1
  };

struct Experience {
  Eigen::VectorXd state;
  TradingAction action;
  double reward;
  Eigen::VectorXd nextState;
  bool done;

  // Metadata
  double confidence;
  std::string timestamp;
  double transactionCost;
  double positionSize;
  double portfolioValue;

  Experience(const Eigen::VectorXd &s,
            TradingAction a,
            double r,
            Eigen::VectorXd &ns,
            bool d)
    : state(s), action(a), reward(r), nextState(ns), done(d),
      confidence(0.0), timestamp(""), transactionCost(0.0),
      positionSize(0.0), portfolioValue(0.0) {}
};

class ReplayBuffer {
public:
  /**
     * @brief Constructs a replay buffer with specified capacity
     * @param capacity Maximum number of experiences to store
     * @throws std::invalid_argument if capacity is 0
     */

  explicit ReplayBuffer(size_t capacity);
    /**
     * @brief Adds a new experience to the buffer
     * @param experience The experience to add
     */
  // Core operations
  void addExperience(const Experience& experience);
    /**
     * @brief Samples a random batch of experiences
     * @param batchSize Number of experiences to sample
     * @return Vector of randomly sampled experiences
     * @throws std::runtime_error if batchSize > buffer size
     */
  std::vector<Experience> sample(size_t batchSize);

  // Utility Methods
  [[nodiscard]] size_t size() const;
  [[nodiscard]] bool isEmpty() const;
  [[nodiscard]] size_t capacity() const;
  [[nodiscard]] double getFillRatio() const;
  void clear();

 private:
   std::deque<Experience> m_buffer;
   size_t m_capacity;
   std::mt19937 m_rng;
};



#endif //REPLAYBUFFER_H
