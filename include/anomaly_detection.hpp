/*
 * AnomalyDetection.cpp v0.2.1
 * https://github.com/ankane/AnomalyDetection.cpp
 * GPL-3.0-or-later License
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>

#include "dist.h"
#include "stl.hpp"

namespace anomaly_detection {

/// The direction to detect anomalies.
enum class Direction {
    /// Positive direction.
    Positive,
    /// Negative direction.
    Negative,
    /// Both directions.
    Both
};

namespace detail {

template<typename T>
T median_sorted(const std::vector<T>& sorted) {
    return (sorted.at((sorted.size() - 1) / 2) + sorted.at(sorted.size() / 2)) / static_cast<T>(2.0);
}

template<typename T>
T median(std::span<const T> data) {
    std::vector<T> sorted(data.begin(), data.end());
    std::ranges::sort(sorted);
    return median_sorted(sorted);
}

template<typename T>
T mad(const std::vector<T>& data, T med) {
    std::vector<T> res;
    res.reserve(data.size());
    for (auto v : data) {
        res.push_back(std::abs(v - med));
    }
    std::ranges::sort(res);
    return static_cast<T>(1.4826) * median_sorted(res);
}

template<typename T>
std::vector<size_t> detect_anoms(std::span<const T> data, size_t num_obs_per_period, float k, float alpha, bool one_tail, bool upper_tail, bool verbose, std::function<void()> callback) {
    size_t n = data.size();

    // Check to make sure we have at least two periods worth of data for anomaly context
    if (n < num_obs_per_period * 2) {
        throw std::invalid_argument("series must contain at least 2 periods");
    }

    // Handle NANs
    size_t nan = std::count_if(data.begin(), data.end(), [](const auto& value) {
        return std::isnan(value);
    });
    if (nan > 0) {
        throw std::invalid_argument("series contains NANs");
    }

    std::vector<T> data2;
    data2.reserve(n);
    T med = median(data);

    if (num_obs_per_period > 1) {
        // Decompose data. This returns a univarite remainder which will be used for anomaly detection. Optionally, we might NOT decompose.
        auto data_decomp = stl::params().robust(true).seasonal_length(data.size() * 10 + 1).fit(data, num_obs_per_period);
        auto seasonal = data_decomp.seasonal;

        for (size_t i = 0; i < n; i++) {
            data2.push_back(data[i] - seasonal.at(i) - med);
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            data2.push_back(data[i] - med);
        }
    }

    size_t num_anoms = 0;
    auto max_outliers = static_cast<size_t>(static_cast<float>(n) * k);
    std::vector<size_t> anomalies;
    anomalies.reserve(max_outliers);

    // Sort data for fast median
    // Use stable sort for indexes for deterministic results
    std::vector<size_t> indexes(n);
    std::iota(indexes.begin(), indexes.end(), 0);
    std::stable_sort(indexes.begin(), indexes.end(), [&data2](size_t a, size_t b) {
        return data2.at(a) < data2.at(b);
    });
    std::ranges::sort(data2);

    // Compute test statistic until r=max_outliers values have been removed from the sample
    for (size_t i = 1; i <= max_outliers; i++) {
        if (verbose) {
            std::cout << i << " / " << max_outliers << " completed" << std::endl;
        }

        // TODO Improve performance between loop iterations
        T ma = median_sorted(data2);
        std::vector<T> ares;
        ares.reserve(data2.size());
        if (one_tail) {
            if (upper_tail) {
                for (auto v : data2) {
                    ares.push_back(v - ma);
                }
            } else {
                for (auto v : data2) {
                    ares.push_back(ma - v);
                }
            }
        } else {
            for (auto v : data2) {
                ares.push_back(std::abs(v - ma));
            }
        }

        // Protect against constant time series
        T data_sigma = mad(data2, ma);
        if (data_sigma == 0.0) {
            break;
        }

        auto iter = std::max_element(ares.begin(), ares.end());
        size_t r_idx_i = std::distance(ares.begin(), iter);

        // Only need to take sigma of r for performance
        T r = ares.at(r_idx_i) / data_sigma;

        anomalies.push_back(indexes.at(r_idx_i));
        data2.erase(data2.begin() + r_idx_i);
        indexes.erase(indexes.begin() + r_idx_i);

        // Compute critical value
        double p;
        if (one_tail) {
            p = 1.0 - alpha / static_cast<double>(n - i + 1);
        } else {
            p = 1.0 - alpha / (2.0 * static_cast<double>(n - i + 1));
        }

        double t = students_t_ppf(p, static_cast<double>(n - i - 1));
        double lam = t * static_cast<double>(n - i) / std::sqrt((static_cast<double>(n - i - 1) + t * t) * static_cast<double>(n - i + 1));

        if (r > lam) {
            num_anoms = i;
        }

        if (callback != nullptr) {
            callback();
        }
    }

    anomalies.resize(num_anoms);

    // Sort like R version
    std::ranges::sort(anomalies);

    return anomalies;
}

} // namespace detail

/// An anomaly detection result.
class AnomalyDetectionResult {
  public:
    /// Returns the anomalies.
    std::vector<size_t> anomalies;
};

/// A set of anomaly detection parameters.
class AnomalyDetectionParams {
    float alpha_ = 0.05f;
    float max_anoms_ = 0.1f;
    Direction direction_ = Direction::Both;
    bool verbose_ = false;
    std::function<void()> callback_ = nullptr;

  public:
    /// Sets the level of statistical significance.
    inline AnomalyDetectionParams alpha(float alpha) {
        this->alpha_ = alpha;
        return *this;
    }

    /// Sets the maximum number of anomalies as percent of data.
    inline AnomalyDetectionParams max_anoms(float max_anoms) {
        this->max_anoms_ = max_anoms;
        return *this;
    }

    /// Sets the direction.
    inline AnomalyDetectionParams direction(Direction direction) {
        this->direction_ = direction;
        return *this;
    }

    /// Sets whether to show progress.
    inline AnomalyDetectionParams verbose(bool verbose) {
        this->verbose_ = verbose;
        return *this;
    }

    /// Sets a callback for each iteration.
    inline AnomalyDetectionParams callback(std::function<void()> callback) {
        this->callback_ = callback;
        return *this;
    }

    /// Detects anomalies in a time series from a span.
    template<typename T>
    inline AnomalyDetectionResult fit(std::span<const T> series, size_t period) const {
        bool one_tail = this->direction_ != Direction::Both;
        bool upper_tail = this->direction_ == Direction::Positive;

        auto anomalies = detail::detect_anoms(series, period, this->max_anoms_, this->alpha_, one_tail, upper_tail, this->verbose_, this->callback_);
        // TODO move
        return AnomalyDetectionResult { anomalies };
    }

    /// Detects anomalies in a time series from a vector.
    template<typename T>
    inline AnomalyDetectionResult fit(const std::vector<T>& series, size_t period) const {
        return fit(std::span<const T>{series}, period);
    }
};

/// Creates a new set of parameters.
inline AnomalyDetectionParams params() {
    return AnomalyDetectionParams{};
}

} // namespace anomaly_detection
