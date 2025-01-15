cmake_minimum_required(VERSION 3.18)

project(anomaly_detection VERSION 0.1.4 LANGUAGES CXX)

include(FetchContent)
include(GNUInstallDirs)

FetchContent_Declare(dist GIT_REPOSITORY https://github.com/ankane/dist.h.git GIT_TAG v0.3.1)
FetchContent_Declare(stl GIT_REPOSITORY https://github.com/ankane/stl-cpp.git GIT_TAG v0.1.5)
FetchContent_MakeAvailable(dist stl)

add_library(anomaly_detection INTERFACE)
add_library(anomaly_detection::anomaly_detection ALIAS anomaly_detection)

target_link_libraries(anomaly_detection PRIVATE dist::dist stl::stl)

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
