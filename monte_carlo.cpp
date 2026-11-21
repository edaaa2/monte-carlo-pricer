#include "monte_carlo.hpp"
#include "monte_carlo_core.hpp"

#include <random>
#include <stdexcept>
#include <omp.h>

MCResult mc_price_serial(const OptionParams& p, long num_paths) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    monte_carlo_detail::PayoffTotals totals{num_paths};
    for (long i = 0; i < num_paths; ++i) {
        const double payoff = monte_carlo_detail::call_payoff(p, dist(gen));
        totals.sum += payoff;
        totals.sum_sq += payoff * payoff;
    }

    return monte_carlo_detail::price_from_totals(p, totals);
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
        unsigned int seed = 42u + static_cast<unsigned int>(tid);

        std::mt19937 gen(seed);
        std::normal_distribution<double> dist(0.0, 1.0);

        #pragma omp for reduction(+:sum, sum_sq)
        for (long i = 0; i < num_paths; ++i) {
            const double payoff = monte_carlo_detail::call_payoff(p, dist(gen));
            sum += payoff;
            sum_sq += payoff * payoff;
        }
    }

    return monte_carlo_detail::price_from_totals(p, {num_paths, sum, sum_sq});
}
