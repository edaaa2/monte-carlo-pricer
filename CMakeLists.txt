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

option(MC_PRICER_BUILD_MPI "Build the pure MPI Monte Carlo pricer" OFF)
option(MC_PRICER_BUILD_HYBRID "Build the MPI and OpenMP Monte Carlo pricer" OFF)
if(MC_PRICER_BUILD_MPI OR MC_PRICER_BUILD_HYBRID)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

if(MC_PRICER_BUILD_MPI)
  add_executable(mc_pricer_mpi main_mpi.cpp black_scholes.cpp)
  target_link_libraries(mc_pricer_mpi PRIVATE MPI::MPI_CXX)
endif()

if(MC_PRICER_BUILD_HYBRID)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  add_executable(mc_pricer_hybrid main_mpi.cpp black_scholes.cpp)
  target_compile_definitions(mc_pricer_hybrid PRIVATE MC_PRICER_HYBRID)
  target_link_libraries(mc_pricer_hybrid PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
endif()
