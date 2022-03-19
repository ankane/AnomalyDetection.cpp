#include <cassert>
#include <vector>

#include "../include/anomaly_detection.hpp"

using anomaly_detection::Direction;

#define ASSERT_EXCEPTION(code, type, message) { \
    try {                                       \
        code;                                   \
        assert(false);                          \
    } catch (const type &e) {                   \
        assert(strcmp(e.what(), message) == 0); \
    }                                           \
}

std::vector<float> generate_series() {
    std::vector<float> series = {
        5.0, 9.0, 2.0, 9.0, 0.0, 6.0, 3.0, 8.0, 5.0, 18.0,
        7.0, 8.0, 8.0, 0.0, 2.0, -5.0, 0.0, 5.0, 6.0, 7.0,
        3.0, 6.0, 1.0, 4.0, 4.0, 4.0, 30.0, 7.0, 5.0, 8.0
    };
    return series;
}

void test_works() {
    auto series = generate_series();
    std::vector<size_t> expected = {9, 15, 26};
    auto res = anomaly_detection::params().max_anoms(0.2).fit(series, 7);
    assert(res.anomalies == expected);
}

void test_direction_pos() {
    auto series = generate_series();
    auto res = anomaly_detection::params()
        .max_anoms(0.2)
        .direction(Direction::Positive)
        .fit(series, 7);
    std::vector<size_t> expected = {9, 26};
    assert(res.anomalies == expected);
}

void test_direction_neg() {
    auto series = generate_series();
    auto res = anomaly_detection::params()
        .max_anoms(0.2)
        .direction(Direction::Negative)
        .fit(series, 7);
    std::vector<size_t> expected = {15};
    assert(res.anomalies == expected);
}

void test_alpha() {
    auto series = generate_series();
    auto res = anomaly_detection::params().max_anoms(0.2).alpha(0.5).fit(series, 7);
    std::vector<size_t> expected = {1, 4, 9, 15, 26};
    assert(res.anomalies == expected);
}

void test_nan() {
    std::vector<float> series(30, 1.0);
    series[15] = NAN;
    ASSERT_EXCEPTION(
        anomaly_detection::params().fit(series, 7),
        std::invalid_argument,
        "series contains NANs"
    );
}

void test_empty_data() {
    std::vector<float> series;
    ASSERT_EXCEPTION(
        anomaly_detection::params().fit(series, 7),
        std::invalid_argument,
        "series must contain at least 2 periods"
    );
}

void test_max_anoms_zero() {
    auto series = generate_series();
    auto res = anomaly_detection::params().max_anoms(0.0).fit(series, 7);
    assert(res.anomalies.empty());
}

void test_callback() {
    auto series = generate_series();
    auto count = 0;
    auto callback = [&count]() { count++; };
    auto res = anomaly_detection::params().callback(callback).fit(series, 7);
    assert(count == 3);
}

int main() {
    test_works();
    test_direction_pos();
    test_direction_neg();
    test_alpha();
    test_nan();
    test_empty_data();
    test_max_anoms_zero();
    test_callback();
    return 0;
}
