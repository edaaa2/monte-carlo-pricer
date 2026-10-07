#pragma once


struct OptionParams {
    double S0;     
    double K;      
    double r;      
    double sigma;  
    double T;      
};

double black_scholes_call_price(const OptionParams& p);
