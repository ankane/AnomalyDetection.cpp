cmake_minimum_required(VERSION 3.21)

project(anomaly_detection VERSION 0.2.1 LANGUAGES CXX)

include(FetchContent)
include(GNUInstallDirs)

FetchContent_Declare(dist GIT_REPOSITORY https://github.com/ankane/dist.h.git GIT_TAG v0.3.1)
FetchContent_Declare(stl GIT_REPOSITORY https://github.com/ankane/stl-cpp.git GIT_TAG v0.2.0)
FetchContent_MakeAvailable(dist stl)

add_library(anomaly_detection INTERFACE)
add_library(anomaly_detection::anomaly_detection ALIAS anomaly_detection)

target_link_libraries(anomaly_detection INTERFACE dist::dist stl::stl)

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

if(PROJECT_IS_TOP_LEVEL)
    option(BUILD_TESTING "" ON)
    option(BUILD_EXAMPLE "" ON)

    if(BUILD_TESTING)
        add_executable(test test/main.cpp)
        target_link_libraries(test PRIVATE anomaly_detection::anomaly_detection)
        if(NOT MSVC)
            target_compile_options(test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        endif()
    endif()

    if(BUILD_EXAMPLE)
        add_executable(example example/main.cpp)
        target_link_libraries(example PRIVATE anomaly_detection::anomaly_detection)
    endif()
endif()
