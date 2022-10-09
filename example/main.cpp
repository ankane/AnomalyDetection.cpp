#include <iostream>

#include "anomaly_detection.hpp"

int main() {
    std::vector<float> series = {
        5.0, 9.0, 2.0, 9.0, 0.0, 6.0, 3.0, 8.0, 5.0, 18.0,
        7.0, 8.0, 8.0, 0.0, 2.0, 15.0, 0.0, 5.0, 6.0, 7.0,
        3.0, 6.0, 1.0, 4.0, 4.0, 4.0, 30.0, 7.0, 5.0, 8.0
    };
    size_t period = 7; // number of observations in a single period

    auto res = anomaly_detection::params().fit(series, period);
    for (auto anomaly : res.anomalies) {
        std::cout << anomaly << std::endl;
    }

    return 0;
}
