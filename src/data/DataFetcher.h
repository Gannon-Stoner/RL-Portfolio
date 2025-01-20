#pragma once

#include <string>
#include <vector>

/**
 * DataFetcher class for retrieving CSV stock data from Alpha Vantage (Daily EOD).
 */
class DataFetcher {
public:
    /**
     * @param apiKey Your Alpha Vantage API key (e.g., "V21B7SHDIVN7RVAD").
     */
    DataFetcher(const std::string& apiKey);

    /**
     * @brief Fetch daily EOD CSV data for a given symbol.
     *
     * @param symbol The stock symbol, e.g., "AAPL", "IBM", etc.
     * @param outputSize Either "compact" (most recent ~100 data points) or "full" (5+ years).
     * @param outData A 2D vector of strings, where each inner vector represents:
     *        [timestamp, open, high, low, close, volume].
     * @return True if successful, false otherwise.
     */
    bool fetchDailyData(const std::string& symbol,
                        const std::string& outputSize,
                        std::vector<std::vector<std::string>>& outData);

private:
    std::string m_apiKey;

    /**
     * @brief Performs HTTP GET request to the given URL, returns response body as a string.
     */
    std::string httpGet(const std::string& url);

    /**
     * @brief Static callback required by libcurl to write chunks of data into a std::string.
     */
    static size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp);
};
