#pragma once
#include "black_scholes.hpp"

struct MCResult {
    double price;
    double standard_error;
    double confidence_low;
    double confidence_high;
}; 


MCResult mc_price_serial(const OptionParams& p, long num_paths);


MCResult mc_price_parallel(const OptionParams& p, long num_paths, int num_threads);
