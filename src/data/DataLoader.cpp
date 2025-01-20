//
// Created by grs on 1/14/25.
//

#include "DataLoader.h"
#include <fstream>
#include <set>
#include <sstream>
#include <spdlog/spdlog.h>


// In DataLoader.cpp

std::vector<EODDataRow> DataLoader::loadData(const std::string& filePath) {
    std::vector<EODDataRow> dataRows;

    std::ifstream file(filePath);
    if (!file.is_open()) {
        spdlog::error("DataLoader: Failed to open file {}", filePath);
        return dataRows;
    }

    spdlog::info("DataLoader: Reading file: {}", filePath);

    std::string line;
    bool isHeader = true;

    while (std::getline(file, line)) {
        if (isHeader) {
            isHeader = false;
            continue;
        }
        if (line.empty()) {
            continue;
        }
        std::stringstream lineStream(line);
        std::string token;
        std::vector<std::string> rowTokens;

        while (std::getline(lineStream, token, ',')) {
            rowTokens.push_back(token);
        }

        // We expect 6 columns
        if (rowTokens.size() != 6) {
            spdlog::warn("DataLoader: Skipping malformed line: {}", line);
            continue;
        }

        // Parse each token
        EODDataRow eod;
        eod.date = rowTokens[0]; // keep string date
        try {
            eod.open = std::stod(rowTokens[1]);
            eod.high = std::stod(rowTokens[2]);
            eod.low = std::stod(rowTokens[3]);
            eod.close = std::stod(rowTokens[4]);
            eod.volume = std::stod(rowTokens[5]);
        } catch (const std::exception& e) {
            spdlog::warn("DataLoader: Conversion error on line: {}. Error: {}", line, e.what());
            continue;
        }

        dataRows.push_back(eod);
    }

    file.close();

    // Sort data by date in ascending order
    std::sort(dataRows.begin(), dataRows.end(),
              [](const EODDataRow& a, const EODDataRow& b) {
                  return a.date < b.date;
              });

    // Validate the sorted data
    if (!validateData(dataRows)) {
        spdlog::error("DataLoader: Data validation failed for {}", filePath);
        return std::vector<EODDataRow>(); // Return empty vector if validation fails
    }

    spdlog::info("DataLoader: Successfully loaded and validated {} rows from {}",
                 dataRows.size(), filePath);
    return dataRows;
}

bool DataLoader::validateData(const std::vector<EODDataRow>& data) {
    if (data.empty()) {
        spdlog::error("Data validation failed: Empty dataset");
        return false;
    }

    // Check for chronological order
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i].date <= data[i-1].date) {
            spdlog::error("Data validation failed: Dates not in chronological order at index {}", i);
            return false;
        }
    }

    // Check for invalid values
    for (size_t i = 0; i < data.size(); ++i) {
        const auto& row = data[i];

        // Check for negative values
        if (row.open <= 0 || row.high <= 0 || row.low <= 0 ||
            row.close <= 0 || row.volume < 0) {
            spdlog::error("Data validation failed: Invalid values at date {}", row.date);
            return false;
        }

        // Check price consistency
        if (row.low > row.high || row.open > row.high || row.open < row.low ||
            row.close > row.high || row.close < row.low) {
            spdlog::error("Data validation failed: Price consistency error at date {}", row.date);
            return false;
        }
    }

    return true;
}

void DataLoader::loadMultipleAssets(const std::vector<std::string> &filePaths) {
    asset_data_.clear();
    aligned_dates_.clear();

    // Load each file
    for (const auto& path : filePaths) {
        std::string asset_name = path.substr(path.find_last_of("/\\") + 1);
        asset_data_[asset_name] = loadData(path);
    }

    alignDates();
}

void DataLoader::alignDates() {
    std::set<std::string> common_dates;
    bool first = true;

    for (const auto& [asset, data] : asset_data_) {
        std::set<std::string> asset_dates;
        for (const auto& row : data) {
            asset_dates.insert(row.date);
        }

        if (first) {
            common_dates = asset_dates;
            first = false;
        } else {
            std::set<std::string> intersection;
            std::set_intersection(
                common_dates.begin(), common_dates.end(),
                asset_dates.begin(), asset_dates.end(),
                std::inserter(intersection, intersection.begin())
                );
            common_dates = intersection;
        }
    }
    aligned_dates_ = std::vector<std::string>(common_dates.begin(), common_dates.end());
    std::sort(aligned_dates_.begin(), aligned_dates_.end());
}

std::vector<Eigen::VectorXd> DataLoader::getPrices() const {
    std::vector<Eigen::VectorXd> price_history;
    size_t num_assets = asset_data_.size();

    for (const auto& date : aligned_dates_) {
        Eigen::VectorXd prices(num_assets);
        int i = 0;

        // Get price for each asset on this date
        for (const auto& [asset, data] : asset_data_) {
            auto it = std::find_if(data.begin(), data.end(), [&date] (const EODDataRow& row) {
                return row.date == date;
            });
            if (it != data.end()) { // end() means not found
                prices(i) = it->close; // Using closing prices
            } else {
                spdlog::warn("Missing data for {} on date {}", asset, date);
                prices(i) = 0.0;
            }
            i++;
        }
        price_history.push_back(prices);
    }
    return price_history;
}

std::vector<Eigen::VectorXd> DataLoader::getVolumes() const {
    std::vector<Eigen::VectorXd> volume_history;
    size_t num_assets = asset_data_.size();

    for (const auto& date : aligned_dates_) {
        Eigen::VectorXd volumes(num_assets);
        int i = 0;

        // Get volume for each asset on this date
        for (const auto& [asset, data] : asset_data_) {
            auto it = std::find_if(data.begin(), data.end(), [&date] (const EODDataRow& row) {
                return row.date == date;
            });
            if (it != data.end()) {
                // Normalize volume by dividing by 1M to keep numbers manageable
                volumes(i) = it->volume / 1000000.0;
            } else {
                spdlog::warn("Missing volume data for {} on date {}", asset, date);
                volumes(i) = 0.0;
            }
            i++;
        }
        volume_history.push_back(volumes);
    }
    return volume_history;
}