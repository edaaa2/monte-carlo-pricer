cmake_minimum_required(VERSION 3.15)
project(monte_carlo_pricer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


find_package(OpenMP)

add_executable(mc_pricer
  main.cpp
  black_scholes.cpp
  monte_carlo.cpp
)

target_include_directories(mc_pricer PRIVATE include)

if(OpenMP_CXX_FOUND)
  target_link_libraries(mc_pricer PRIVATE OpenMP::OpenMP_CXX)
endif()
