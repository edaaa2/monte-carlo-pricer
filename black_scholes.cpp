#include "black_scholes.hpp"
#include <cmath>


static double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double black_scholes_call_price(const OptionParams& p) {
    double d1 = (std::log(p.S0 / p.K) + (p.r + 0.5 * p.sigma * p.sigma) * p.T)
                / (p.sigma * std::sqrt(p.T));
    double d2 = d1 - p.sigma * std::sqrt(p.T);

    return p.S0 * norm_cdf(d1) - p.K * std::exp(-p.r * p.T) * norm_cdf(d2);
}
