//
// Created by grs on 1/15/25.
//

#include "ReplayBuffer.h"

ReplayBuffer::ReplayBuffer(size_t capacity)
    : m_capacity(capacity), m_rng(std::random_device{}())
{
    if (capacity == 0) {
        throw std::invalid_argument("Replay buffer capacity must be greater than 0");
    }
}

void ReplayBuffer::addExperience(const Experience &experience) {
    if (m_buffer.size() >= m_capacity) {
        m_buffer.pop_front(); //Remove from front of the deque (the oldest)
    }
    //Add new experience to the back of the buffer
    m_buffer.push_back(experience);
}

std::vector<Experience> ReplayBuffer::sample(size_t batchSize) {
    if (batchSize > m_buffer.size()) {
        throw std::runtime_error("Requested batch size is greater than buffer size");
    }

    // Vector stores our sampled experiences
    std::vector<Experience> batch;
    batch.reserve(batchSize);

    // Creates the indices for sampling
    std::vector<size_t> indices(m_buffer.size());
    for (size_t i = 0; i < m_buffer.size(); ++i) {
        indices[i] = i;
    }

    // Shuffle indices
    std::shuffle(indices.begin(), indices.end(), m_rng);

    for (size_t i = 0; i < batchSize; ++i) {
        batch.push_back(m_buffer[indices[i]]);
    }

    return batch;
}

size_t ReplayBuffer::size() const {
    return m_buffer.size();
}

bool ReplayBuffer::isEmpty() const {
    return m_buffer.empty();
}



