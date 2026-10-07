#include "black_scholes.hpp"
#include "monte_carlo.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>

int main() {

    OptionParams params{
        /*S0=*/100.0,
        /*K=*/100.0,
        /*r=*/0.05,
        /*sigma=*/0.2,
        /*T=*/1.0
    };

    // Analytical benchmark
    double bs_price = black_scholes_call_price(params);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Black-Scholes analytical price: " << bs_price << "\n\n";



    // Serial Monte Carlo experiments
    for (long num_paths : {10000L, 100000L, 1000000L}) {
        MCResult result = mc_price_serial(params, num_paths);
        double error_pct = std::abs(result.price - bs_price) / bs_price * 100.0;

        std::cout << "N = " << num_paths
                  << "  Price = " << result.price
                  << "  Standard Error = " << result.standard_error
                  << "  Error vs BS = " << error_pct << "%\n";

        std::cout << "    95% CI = ["
                  << result.confidence_low
                  << ", "
                  << result.confidence_high
                  << "]\n\n";

    }
    

    //parallel
    long num_paths = 1000000L;

    for (int num_threads : {1, 2, 4, 8}) {

        MCResult result =
            mc_price_parallel(params, num_paths, num_threads);

        double error_pct =
            std::abs(result.price - bs_price)
            / bs_price * 100.0;

        std::cout << "Threads = " << num_threads
                  << "  N = " << num_paths
                  << "  Price = " << result.price
                  << "  Standard Error = " << result.standard_error
                  << "  Error vs BS = " << error_pct << "%\n";

        std::cout << "    95% CI = ["
                  << result.confidence_low
                  << ", "
                  << result.confidence_high
                  << "]\n\n";
    }

        // Performance Benchmark
    std::cout << "\n Performance Benchmark :\n";

    long benchmark_paths = 10000000L;

    // Serial baseline
    double serial_start = omp_get_wtime();

    mc_price_serial(params, benchmark_paths);

    double serial_end = omp_get_wtime();
    double serial_time = serial_end - serial_start;

    std::cout << "Serial time = "
              << serial_time << " s\n\n";

    // Parallel benchmark
    for (int num_threads : {1, 2, 4, 8}) {

        double start = omp_get_wtime();

        mc_price_parallel(params, benchmark_paths, num_threads);

        double end = omp_get_wtime();
        double parallel_time = end - start;

        double speedup = serial_time / parallel_time;
        double efficiency = speedup / num_threads * 100.0;

        std::cout << "Threads = " << num_threads
                  << "  Time = " << parallel_time << " s"
                  << "  Speedup = " << speedup << "x"
                  << "  Efficiency = " << efficiency << "%\n";
    }

    return 0;
}
