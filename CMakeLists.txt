cmake_minimum_required(VERSION 3.18)

project(anomaly_detection VERSION 0.1.4 LANGUAGES CXX)

include(GNUInstallDirs)

add_library(anomaly_detection INTERFACE)
add_library(anomaly_detection::anomaly_detection ALIAS anomaly_detection)

target_compile_features(anomaly_detection INTERFACE cxx_std_17)

target_include_directories(
    anomaly_detection
    INTERFACE
    "${PROJECT_SOURCE_DIR}/include/"
)

install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
