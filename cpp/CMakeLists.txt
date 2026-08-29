cmake_minimum_required(VERSION 3.16)
project(navsim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 QUIET)
if(NOT EIGEN3_FOUND)
  # Homebrew Apple Silicon default location
  if(EXISTS "/opt/homebrew/include/eigen3")
    set(EIGEN3_INCLUDE_DIR "/opt/homebrew/include/eigen3")
    set(EIGEN3_FOUND TRUE)
    message(STATUS "Eigen3 found at /opt/homebrew/include/eigen3")
  else()
    message(FATAL_ERROR "Eigen3 not found (brew install eigen)")
  endif()
endif()

add_library(nav INTERFACE)
target_include_directories(nav INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})
target_compile_options(nav INTERFACE -O2)

add_executable(eskf_demo examples/eskf_demo.cpp)
target_link_libraries(eskf_demo PRIVATE nav)

add_executable(test_nav tests/test_nav.cpp)
target_link_libraries(test_nav PRIVATE nav)
enable_testing()
add_test(NAME nav COMMAND test_nav)
