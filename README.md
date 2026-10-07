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
