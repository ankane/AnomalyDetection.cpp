#include <cassert>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <anomaly_detection.hpp>

using anomaly_detection::AnomalyDetectionResult;
using anomaly_detection::Direction;

template<typename T>
void assert_exception(
    const std::function<void(void)>& code,
    std::optional<std::string_view> message = std::nullopt
) {
    std::optional<T> exception;
    try {
        code();
    } catch (const T& e) {
        exception = e;
    }
    assert(exception.has_value());
    if (message) {
        assert(std::string_view{exception.value().what()} == message.value());
    }
}

template<typename T>
std::vector<T> generate_series() {
    std::vector<T> series{
        5.0, 9.0, 2.0, 9.0, 0.0, 6.0, 3.0, 8.0, 5.0, 18.0,
        7.0, 8.0, 8.0, 0.0, 2.0, -5.0, 0.0, 5.0, 6.0, 7.0,
        3.0, 6.0, 1.0, 4.0, 4.0, 4.0, 30.0, 7.0, 5.0, 8.0
    };
    return series;
}

template<typename T>
void test_works() {
    std::vector<T> series = generate_series<T>();
    std::vector<size_t> expected{9, 15, 26};
    AnomalyDetectionResult res = anomaly_detection::params().max_anoms(0.2f).fit(series, 7);
    assert(res.anomalies == expected);
}

template<typename T>
void test_span() {
    std::vector<T> series = generate_series<T>();
    std::vector<size_t> expected{9, 15, 26};
    AnomalyDetectionResult res = anomaly_detection::params().max_anoms(0.2f).fit(std::span<const T>(series), 7);
    assert(res.anomalies == expected);
}

template<typename T>
void test_no_seasonality() {
    std::vector<T> series{1.0, 6.0, 2.0, 3.0, 3.0, 0.0};
    std::vector<size_t> expected{1};
    AnomalyDetectionResult res = anomaly_detection::params().max_anoms(0.2f).fit(series, 1);
    assert(res.anomalies == expected);
}

template<typename T>
void test_direction_pos() {
    std::vector<T> series = generate_series<T>();
    AnomalyDetectionResult res = anomaly_detection::params()
        .max_anoms(0.2f)
        .direction(Direction::Positive)
        .fit(series, 7);
    std::vector<size_t> expected{9, 26};
    assert(res.anomalies == expected);
}

template<typename T>
void test_direction_neg() {
    std::vector<T> series = generate_series<T>();
    AnomalyDetectionResult res = anomaly_detection::params()
        .max_anoms(0.2f)
        .direction(Direction::Negative)
        .fit(series, 7);
    std::vector<size_t> expected{15};
    assert(res.anomalies == expected);
}

template<typename T>
void test_alpha() {
    std::vector<T> series = generate_series<T>();
    AnomalyDetectionResult res = anomaly_detection::params().max_anoms(0.2f).alpha(0.5).fit(series, 7);
    std::vector<size_t> expected{1, 4, 9, 15, 26};
    assert(res.anomalies == expected);
}

template<typename T>
void test_nan() {
    std::vector<T> series(30, 1.0);
    series.at(15) = std::numeric_limits<T>::quiet_NaN();
    assert_exception<std::invalid_argument>([&]() {
        anomaly_detection::params().fit(series, 7);
    }, "series contains NANs");
}

template<typename T>
void test_empty_data() {
    std::vector<T> series;
    assert_exception<std::invalid_argument>([&]() {
        anomaly_detection::params().fit(series, 7);
    }, "series must contain at least 2 periods");
}

template<typename T>
void test_max_anoms_zero() {
    std::vector<T> series = generate_series<T>();
    AnomalyDetectionResult res = anomaly_detection::params().max_anoms(0.0f).fit(series, 7);
    assert(res.anomalies.empty());
}

template<typename T>
void test_callback() {
    std::vector<T> series = generate_series<T>();
    size_t count = 0;
    auto callback = [&count]() { count++; };
    AnomalyDetectionResult res = anomaly_detection::params().callback(callback).fit(series, 7);
    assert(count == 3);
}

template<typename T>
void test_type() {
    test_works<T>();
    test_span<T>();
    test_no_seasonality<T>();
    test_direction_pos<T>();
    test_direction_neg<T>();
    test_alpha<T>();
    test_nan<T>();
    test_empty_data<T>();
    test_max_anoms_zero<T>();
    test_callback<T>();
}

int main() {
    test_type<float>();
    test_type<double>();
    return 0;
}
