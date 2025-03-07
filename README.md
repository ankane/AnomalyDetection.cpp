# AnomalyDetection.cpp

Time series [AnomalyDetection](https://github.com/twitter/AnomalyDetection) for C++

Learn [how it works](https://blog.twitter.com/engineering/en_us/a/2015/introducing-practical-and-robust-anomaly-detection-in-a-time-series)

[![Build Status](https://github.com/ankane/AnomalyDetection.cpp/actions/workflows/build.yml/badge.svg)](https://github.com/ankane/AnomalyDetection.cpp/actions)

## Installation

Add the headers to your project (supports C++17 and greater).

- [anomaly_detection.hpp](https://raw.githubusercontent.com/ankane/AnomalyDetection.cpp/v0.2.1/include/anomaly_detection.hpp)
- [dist.h](https://raw.githubusercontent.com/ankane/dist.h/v0.3.1/include/dist.h)
- [stl.hpp](https://raw.githubusercontent.com/ankane/stl-cpp/v0.2.0/include/stl.hpp)

There is also support for CMake and FetchContent:

```cmake
include(FetchContent)

FetchContent_Declare(anomaly_detection GIT_REPOSITORY https://github.com/ankane/AnomalyDetection.cpp.git GIT_TAG v0.2.1)
FetchContent_MakeAvailable(anomaly_detection)

target_link_libraries(app PRIVATE anomaly_detection::anomaly_detection)
```

## Getting Started

Include the header

```cpp
#include "anomaly_detection.hpp"
```

Detect anomalies in a time series

```cpp
std::vector<float> series = {
    5.0, 9.0, 2.0, 9.0, 0.0, 6.0, 3.0, 8.0, 5.0, 18.0,
    7.0, 8.0, 8.0, 0.0, 2.0, 15.0, 0.0, 5.0, 6.0, 7.0,
    3.0, 6.0, 1.0, 4.0, 4.0, 4.0, 30.0, 7.0, 5.0, 8.0
};
size_t period = 7; // number of observations in a single period

auto res = anomaly_detection::params().fit(series, period);
```

Get anomalies

```cpp
res.anomalies;
```

## Parameters

Set parameters

```cpp
anomaly_detection::params()
    .alpha(0.05)                    // level of statistical significance
    .max_anoms(0.1)                 // maximum number of anomalies as percent of data
    .direction(Direction::Both)     // Positive, Negative, or Both
    .verbose(false);                // show progress
```

## Credits

This library was ported from the [AnomalyDetection](https://github.com/twitter/AnomalyDetection) R package and is available under the same license. It uses [stl-cpp](https://github.com/ankane/stl-cpp) for seasonal-trend decomposition and [dist.h](https://github.com/ankane/dist.h) for the quantile function.

## References

- [Automatic Anomaly Detection in the Cloud Via Statistical Learning](https://arxiv.org/abs/1704.07706)

## History

View the [changelog](https://github.com/ankane/AnomalyDetection.cpp/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/AnomalyDetection.cpp/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/AnomalyDetection.cpp/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/ankane/AnomalyDetection.cpp.git
cd AnomalyDetection.cpp
cmake -S . -B build
cmake --build build
build/test
```
