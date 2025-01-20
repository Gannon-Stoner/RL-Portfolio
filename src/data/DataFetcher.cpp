#include "DataFetcher.h"
#include <curl/curl.h>
#include <spdlog/spdlog.h>
#include <sstream>

DataFetcher::DataFetcher(const std::string& apiKey)
    : m_apiKey(apiKey) {}

/**
 * fetchDailyData:
 * Build the Alpha Vantage URL for daily data in CSV format, make an HTTP GET,
 * then parse the CSV into outData: each row is [date, open, high, low, close, volume].
 */
bool DataFetcher::fetchDailyData(const std::string& symbol,
                                 const std::string& outputSize,
                                 std::vector<std::vector<std::string>>& outData)
{
    // Example URL:
    // https://www.alphavantage.co/query?function=TIME_SERIES_DAILY
    // &symbol=AAPL
    // &apikey=V21B7SHDIVN7RVAD
    // &datatype=csv
    // &outputsize=compact
    std::string url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
                      "&symbol=" + symbol +
                      "&apikey=" + m_apiKey +
                      "&datatype=csv"
                      "&outputsize=" + outputSize;

    spdlog::info("Fetching daily CSV data for symbol: {}", symbol);
    spdlog::info("URL: {}", url);

    // 1) Perform the HTTP GET
    std::string response = httpGet(url);
    if (response.empty()) {
        spdlog::error("Empty response from Alpha Vantage for symbol: {}", symbol);
        return false;
    }

    // 2) Parse the CSV response
    // CSV columns: timestamp,open,high,low,close,volume
    std::stringstream ss(response);
    std::string line;
    bool isHeader = true;

    while (std::getline(ss, line)) {
        // skip the first line which is the header
        if (isHeader) {
            isHeader = false;
            continue;
        }
        if (line.empty()) {
            continue;
        }

        // tokenize by comma
        std::stringstream lineStream(line);
        std::string token;
        std::vector<std::string> row;

        while (std::getline(lineStream, token, ',')) {
            row.push_back(token);
        }

        // Expect 6 columns: date, open, high, low, close, volume
        if (row.size() == 6) {
            outData.push_back(row);
        }
    }

    spdlog::info("Fetched {} rows for symbol: {}", outData.size(), symbol);
    return !outData.empty();
}

/**
 * httpGet: uses libcurl to make a GET request and return the response body.
 */
std::string DataFetcher::httpGet(const std::string& url) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        spdlog::error("Failed to initialize CURL.");
        return {};
    }

    std::string responseBuffer;

    // Set URL
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    // Follow HTTP redirections (just in case)
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    // Pass callback function
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &DataFetcher::writeCallback);
    // Pass the string to write data into
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBuffer);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        spdlog::error("curl_easy_perform() failed: {}", curl_easy_strerror(res));
        curl_easy_cleanup(curl);
        return {};
    }

    curl_easy_cleanup(curl);
    return responseBuffer;
}

/**
 * writeCallback: static method that libcurl calls when it has data to write.
 * We append the incoming data chunk to the string 'userp'.
 */
size_t DataFetcher::writeCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t totalSize = size * nmemb;
    std::string* str = static_cast<std::string*>(userp);
    str->append((char*)contents, totalSize);
    return totalSize;
}
