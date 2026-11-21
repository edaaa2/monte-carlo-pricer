# Parallel Monte Carlo Option Pricer

A C++17 implementation of European option pricing using Monte Carlo simulation, with the Black-Scholes analytical solution as a benchmark.

The project starts from a serial Monte Carlo implementation and extends it with OpenMP parallelization, thread-local random-number generators, statistical error estimation, and performance benchmarking.

The main goal is to study the trade-off between **correctness, statistical accuracy, and computational performance** in Monte Carlo pricing.

## Features

* European call option pricing using Monte Carlo simulation
* Black-Scholes analytical solution as a correctness benchmark
* Geometric Brownian Motion (GBM) for terminal stock-price simulation
* `std::mt19937` and `std::normal_distribution` for random-number generation
* Sample variance and Monte Carlo standard error
* Approximate 95% confidence interval
* Configurable number of Monte Carlo paths
* Serial and OpenMP parallel implementations
* Thread-local random-number generators
* OpenMP reduction for simulation results
* Runtime benchmarking with different thread counts
* Speedup and parallel efficiency analysis
* C++17 / CMake build system

## Project Structure

```text
monte-carlo-pricer/
├── CMakeLists.txt
├── README.md
├── black_scholes.cpp
├── black_scholes.hpp
├── monte_carlo.cpp
├── monte_carlo.hpp
└── main.cpp
```

## Methodology

For a European call option, the Monte Carlo estimator follows three main steps.

### 1. Simulate the terminal stock price

Under the Black-Scholes model:

```text
S_T = S_0 * exp((r - 0.5 * sigma^2) * T
                + sigma * sqrt(T) * Z)
```

where:

* `S_0` — initial stock price
* `K` — strike price
* `r` — risk-free interest rate
* `sigma` — volatility
* `T` — time to maturity
* `Z` — standard normal random variable

### 2. Calculate the option payoff

For a European call:

```text
payoff = max(S_T - K, 0)
```

### 3. Discount the expected payoff

The Monte Carlo price is estimated as:

```text
price = exp(-r * T) * mean(payoffs)
```

The implementation also estimates the statistical uncertainty using the standard error:

```text
SE = discounted_std_dev(payoffs) / sqrt(N)
```

An approximate 95% confidence interval is calculated as:

```text
price ± 1.96 * SE
```

## Serial Monte Carlo Baseline

The serial implementation was developed first to establish a correctness baseline before introducing parallelization.

### Implemented

* Serial Monte Carlo simulation
* Standard normal random-number generation
* European call payoff calculation
* Discounted Monte Carlo price
* Sample variance
* Standard error
* Approximate 95% confidence interval
* Comparison with the Black-Scholes analytical solution

### Validation

Using:

```text
S0    = 100
K     = 100
r     = 0.05
sigma = 0.20
T     = 1.0
```

The Black-Scholes analytical price is:

```text
10.4506
```

Monte Carlo validation results:

| Paths (N) | Monte Carlo Price | Standard Error | Error vs. BS |
| --------: | ----------------: | -------------: | -----------: |
|    10,000 |           10.4476 |         0.1481 |      0.0283% |
|   100,000 |           10.5233 |         0.0468 |      0.6959% |
| 1,000,000 |           10.4751 |         0.0147 |      0.2349% |

For example, with 1,000,000 simulated paths:

```text
Monte Carlo Price = 10.4751
Standard Error    = 0.0147
95% CI            = [10.4462, 10.5040]
```

The Black-Scholes price of `10.4506` lies inside the 95% confidence interval.

The standard error decreases approximately according to:

```text
SE ∝ 1 / sqrt(N)
```

This demonstrates the statistical convergence behaviour of the Monte Carlo estimator.

## Why Black-Scholes?

Monte Carlo pricing produces a statistical estimate rather than a deterministic result for each run.

The Black-Scholes analytical solution provides a known benchmark for a European call under the same model assumptions.

This makes it possible to validate the Monte Carlo implementation before introducing parallelization and performance optimizations.

## OpenMP Parallelization

Monte Carlo path simulations are naturally parallel because individual simulation paths are largely independent.

The parallel implementation uses:

### Thread-local random-number generators

Each OpenMP thread creates its own:

```cpp
std::mt19937
```

and:

```cpp
std::normal_distribution<double>
```

This avoids sharing a single RNG between threads.

### OpenMP reduction

The accumulated payoff and squared payoff are combined using:

```cpp
#pragma omp for reduction(+:sum, sum_sq)
```

This avoids race conditions when calculating:

* the total payoff
* the sum of squared payoffs

The parallel implementation therefore maintains the statistical calculations required for both the option price and standard error.

## Parallel Correctness

The parallel implementation was tested using 1, 2, 4, and 8 threads with 1,000,000 Monte Carlo paths.

```text
Threads = 1
Price = 10.4682
Standard Error = 0.0148
95% CI = [10.4393, 10.4971]

Threads = 2
Price = 10.4415
Standard Error = 0.0147
95% CI = [10.4126, 10.4703]

Threads = 4
Price = 10.4373
Standard Error = 0.0147
95% CI = [10.4085, 10.4661]

Threads = 8
Price = 10.4368
Standard Error = 0.0147
95% CI = [10.4079, 10.4656]
```

All estimates are close to the Black-Scholes analytical value of `10.4506`, and the confidence intervals are consistent with the analytical benchmark.

The exact Monte Carlo estimate changes between runs and thread counts because different random samples are generated. This is expected for a stochastic simulation.

## Performance Benchmark

The parallel implementation was benchmarked using:

```text
Compiler: GCC
Language: C++17
Optimization: -O3
OpenMP: enabled
Monte Carlo paths: 10,000,000
Threads: 1, 2, 4, 8
Timing: omp_get_wtime()
```

The serial implementation was used as the baseline.

### Results

| Configuration     | Runtime (s) |   Speedup | Parallel Efficiency |
| ----------------- | ----------: | --------: | ------------------: |
| Serial            |      0.3480 |     1.00× |                   — |
| OpenMP, 1 thread  |      0.3360 |     1.04× |             103.57% |
| OpenMP, 2 threads |      0.2050 |     1.70× |              84.88% |
| OpenMP, 4 threads |      0.1230 |     2.83× |              70.73% |
| OpenMP, 8 threads |      0.0690 | **5.04×** |          **63.04%** |

Speedup is calculated as:

```text
Speedup(p) = T_serial / T_parallel(p)
```

Parallel efficiency is calculated as:

```text
Efficiency(p) = Speedup(p) / p
```

The 8-thread implementation achieves approximately **5.04× speedup** compared with the serial baseline, with approximately **63% parallel efficiency**.

The slightly greater-than-100% efficiency measured for the 1-thread OpenMP case is caused by normal runtime measurement variability rather than actual superlinear parallel scaling.

### Performance Interpretation

The scaling is sublinear:

```text
2 threads → 1.70×
4 threads → 2.83×
8 threads → 5.04×
```

This is expected because the workload contains overheads including:

* rando

## Pure MPI and hybrid MPI + OpenMP benchmark

The benchmark ran on a single WSL Ubuntu host with Open MPI, using 10,000,000 total paths per run. Each configuration ran four times with CPU binding disabled. Every run reported the requested total path count.

| Configuration | MPI ranks | Threads per rank | Run times (s) | Median (s) | Speedup vs. single-rank MPI |
| --- | ---: | ---: | --- | ---: | ---: |
| Pure MPI baseline | 1 | 1 | 0.453619, 0.501446, 0.479266, 0.473009 | 0.4761 | 1.00x |
| Pure MPI | 8 | 1 | 0.101337, 0.109836, 0.128808, 0.150255 | 0.1193 | 3.99x |
| Hybrid | 4 | 2 | 0.134962, 0.135972, 0.097311, 0.131172 | 0.1331 | 3.58x |
| Hybrid | 2 | 4 | 0.103208, 0.105964, 0.120026, 0.155530 | 0.1130 | 4.21x |
| Hybrid | 1 | 8 | 0.142010, 0.126716, 0.113159, 0.127749 | 0.1272 | 3.74x |

Each timing is the maximum elapsed time across ranks for RNG initialization, simulation, and reduction of payoff statistics. MPI startup and shutdown are outside the timed region. The two-rank, four-thread configuration had the lowest median. Its observed times overlapped with the eight-rank pure MPI configuration.

To reproduce the comparison in a Linux environment with Open MPI and OpenMP available, build with both targets enabled:

```bash
cmake -S . -B build-bench -DMC_PRICER_BUILD_MPI=ON -DMC_PRICER_BUILD_HYBRID=ON
cmake --build build-bench
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE mpiexec --oversubscribe --bind-to none -n 8 ./build-bench/mc_pricer_mpi 10000000
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE mpiexec --oversubscribe --bind-to none -n 2 ./build-bench/mc_pricer_hybrid 10000000
```

### Phase timings

A follow-up run used the same host, path count, and binding policy. The table reports the median of three runs for each configuration. Each timing is first maximized independently across ranks.

| Configuration | Total (s) | Local simulation (s) | Payoff reduction (s) |
| --- | ---: | ---: | ---: |
| Pure MPI, 8 ranks | 0.121776 | 0.121710 | 0.032949 |
| Hybrid, 2 ranks with 4 threads each | 0.152386 | 0.152276 | 0.000642 |

Local simulation includes RNG initialization and any OpenMP work. Payoff reduction includes time spent waiting for other ranks to reach the collectives. The phase maxima can come from different ranks, so they cannot be added to recover the total. In these runs, the total was close to the slowest local simulation for both configurations.

### Single-thread path-loop profiling

A temporary C++17 microbenchmark compiled with GCC `-O3` on WSL measured 10,000,000 iterations per loop. The table shows the median of five runs.

| Loop | Median (s) |
| --- | ---: |
| Generate and sum standard normal samples | 0.2592 |
| Calculate payoffs and accumulate both totals from pre-generated samples | 0.0722 |
| Generate samples, calculate payoffs, and accumulate both totals | 0.4144 |

The payoff-only and full loops produced matching printed checksums for the same seed. These loops have different memory access and accumulation patterns, so their timings indicate the relative cost of the work and do not sum to the full-loop time.

### Random number engine optimization

The MPI and hybrid targets use `std::mt19937_64` with the existing rank and thread seed scheme. In a single-thread experiment with 10,000,000 paths for each of eight seeds (42–49), full-loop median time fell from 0.4738 s with `std::mt19937` to 0.3889 s with `std::mt19937_64`. All 16 confidence intervals covered the Black–Scholes reference price.

The following WSL comparison alternated the two binaries over four runs per configuration at 10,000,000 total paths, with CPU binding disabled. Times are medians of the maximum rank time.

| Configuration | `mt19937` (s) | `mt19937_64` (s) | Time reduction |
| --- | ---: | ---: | ---: |
| Pure MPI, 8 ranks | 0.102821 | 0.096165 | 6.5% |
| Hybrid, 2 ranks with 4 threads each | 0.119195 | 0.095375 | 20.0% |

The new engine changes the generated samples. Both configurations still reported a standard error of 0.0047 and confidence intervals covering the analytical price. Pure MPI and hybrid with one thread per rank produced identical printed pricing statistics with three ranks and 1,000,001 paths.

### CPU-pinned strong scaling

All configurations priced 10,000,000 paths on one WSL Ubuntu host using GCC 15.2 and Release optimization. Each configuration ran three times. The benchmark used one logical CPU per physical core, with disjoint CPU sets for MPI ranks and OpenMP threads bound within each rank's set.

| Workers | OpenMP threads | OpenMP median (s) | MPI ranks | MPI median (s) | Hybrid ranks × threads | Hybrid median (s) |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 1 | 0.437380 | 1 | 0.391452 | — | — |
| 2 | 2 | 0.234474 | 2 | 0.203713 | 1 × 2 | 0.232752 |
| 4 | 4 | 0.188529 | 4 | 0.145709 | 2 × 2 | 0.174026 |
| 8 | 8 | 0.125309 | 8 | 0.088517 | 2 × 4 | 0.106433 |

OpenMP timing covered one call to the existing pricing function. MPI and hybrid timings used the maximum rank time, including local simulation and payoff reductions. The executables exclude process startup. OpenMP uses `mt19937`, while MPI and hybrid use `mt19937_64`. Eight OpenMP threads achieved 3.49x speedup over one OpenMP thread. Eight MPI ranks achieved 4.42x speedup over one MPI rank, and the eight-worker hybrid configuration achieved 3.68x relative to that MPI baseline. At eight workers the observed ranges overlapped for pure MPI (0.084681–0.129464 s) and hybrid (0.092003–0.128586 s).
