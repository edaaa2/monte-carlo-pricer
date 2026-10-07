#include "monte_carlo.hpp"

#include <random>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <omp.h>

MCResult mc_price_serial(const OptionParams& p, long num_paths) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    
    double sum = 0.0, sum_sq = 0.0;
    for (long i = 0; i < num_paths; ++i) {
        double Z = dist(gen);
        double S_T = p.S0 * std::exp((p.r - 0.5*p.sigma*p.sigma)*p.T + p.sigma*std::sqrt(p.T)*Z);      // 几何布朗运动公式
        double payoff = std::max(S_T - p.K, 0.0);
        sum += payoff;
        sum_sq += payoff * payoff;
    }
    double average_payoff = sum/num_paths;  
    
    double variance = (sum_sq - num_paths * average_payoff * average_payoff)/ (num_paths - 1);
    
    double price =std::exp(-p.r * p.T) * sum / num_paths;

    double standard_deviation = std::sqrt(variance);
    double standard_error =std::exp(-p.r * p.T)*standard_deviation / std::sqrt(num_paths);

    constexpr double z = 1.96;

    double confidence_low =
        price - z * standard_error;

    double confidence_high =
        price + z * standard_error;

    return MCResult{price, standard_error, confidence_low, confidence_high};
}

MCResult mc_price_parallel(const OptionParams& p, long num_paths, int num_threads) {
    
    if (num_paths < 2) {
        throw std::invalid_argument("num_paths must be at least 2");
    }

    if (num_threads < 1) {
        throw std::invalid_argument("num_threads must be at least 1");
    }


    double sum = 0.0;
    double sum_sq = 0.0;


    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        unsigned int seed =
            42u + static_cast<unsigned int>(tid);

        std::mt19937 gen(seed);
        std::normal_distribution<double> dist(0.0, 1.0);

        #pragma omp for reduction(+:sum, sum_sq)
        for (long i = 0; i < num_paths; ++i) {

            double Z = dist(gen);

            double S_T = p.S0 * std::exp(
                (p.r - 0.5 * p.sigma * p.sigma) * p.T
                + p.sigma * std::sqrt(p.T) * Z
            );

            double payoff =
                std::max(S_T - p.K, 0.0);

            sum += payoff;
            sum_sq += payoff * payoff;
        }
    }

    double average_payoff = sum / num_paths;

    double variance =
        (sum_sq - num_paths * average_payoff * average_payoff)
        / (num_paths - 1);

    double price =std::exp(-p.r * p.T) * average_payoff;

    double standard_deviation =std::sqrt(variance);

    double standard_error =std::exp(-p.r * p.T) * standard_deviation / std::sqrt(static_cast<double>(num_paths));

    constexpr double z = 1.96;

    double confidence_low = price - z * standard_error;

    double confidence_high = price + z * standard_error;

    return MCResult{ price, standard_error, confidence_low, confidence_high};
        
}
