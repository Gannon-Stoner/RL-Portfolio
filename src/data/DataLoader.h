//
// Created by grs on 1/14/25.
//

#pragma once

#ifndef DATALOADER_H
#define DATALOADER_H

#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense>

// Struct holds one row of EOD Market Data
struct EODDataRow {
    std::string date;  // "YYYY-MM-DD"
    double open;
    double high;
    double low;
    double close;
    double volume;
};

// Handles reading local CSV files and returns a vector of EODDataRow structs
class DataLoader {
public:
    // Load single file
    std::vector<EODDataRow> loadData(const std::string& filePath);
    // Load multiple CSV files, one for each asset
    void loadMultipleAssets(const std::vector<std::string>& filePaths);

    // Getter Methods
    std::vector<Eigen::VectorXd> getPrices() const;
    std::vector<Eigen::VectorXd> getVolumes() const;  // New method
    std::vector<std::string> getAlignedDates() const { return aligned_dates_; }
    size_t getNumAssets() const { return asset_data_.size(); }

private:
    bool validateData(const std::vector<EODDataRow>& data);
    void alignDates();
    // Storing data in multiple assets
    std::map<std::string, std::vector<EODDataRow>> asset_data_;
    std::vector<std::string> aligned_dates_;

};



#endif //DATALOADER_H
