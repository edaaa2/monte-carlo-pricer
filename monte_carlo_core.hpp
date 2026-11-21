#pragma once

#include "monte_carlo.hpp"

#include <algorithm>
#include <cmath>

namespace monte_carlo_detail {

struct PayoffTotals {
    long path_count = 0;
    double sum = 0.0;
    double sum_sq = 0.0;
};

inline double call_payoff(const OptionParams& p, double standard_normal) {
    const double terminal_price = p.S0 * std::exp(
        (p.r - 0.5 * p.sigma * p.sigma) * p.T
        + p.sigma * std::sqrt(p.T) * standard_normal
    );
    return std::max(terminal_price - p.K, 0.0);
}

inline MCResult price_from_totals(const OptionParams& p, const PayoffTotals& totals) {
    const double mean = totals.sum / totals.path_count;
    const double variance =
        (totals.sum_sq - totals.path_count * mean * mean) / (totals.path_count - 1);
    const double discount = std::exp(-p.r * p.T);
    const double price = discount * mean;
    const double standard_error =
        discount * std::sqrt(variance) / std::sqrt(static_cast<double>(totals.path_count));
    constexpr double z = 1.96;

    return {price, standard_error, price - z * standard_error, price + z * standard_error};
}

}  // namespace monte_carlo_detail
