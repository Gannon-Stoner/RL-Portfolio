cmake_minimum_required(VERSION 3.30)
project(RL_Portfolio)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(RL_Portfolio src/main.cpp
        src/environment/PortfolioEnv.cpp
        src/environment/PortfolioEnv.h
        src/agent/RLAgent.cpp
        src/agent/RLAgent.h
        src/backtester/Backtester.cpp
        src/backtester/Backtester.h
        src/data/DataLoader.cpp
        src/data/DataLoader.h
        src/utils/Logger.cpp
        src/utils/Logger.h
        src/data/DataFetcher.cpp
        src/data/DataFetcher.h
        src/memory/ReplayBuffer.cpp
        src/memory/ReplayBuffer.h
        src/utils/Config.h)

target_link_libraries(RL_Portfolio
    PRIVATE
        Eigen3::Eigen
        spdlog::spdlog
        CURL::libcurl
        nlohmann_json::nlohmann_json)

target_include_directories(RL_Portfolio
        PRIVATE
        ${EIGEN3_INCLUDE_DIRS}
)