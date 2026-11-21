#include "monte_carlo_core.hpp"

#include <mpi.h>

#include <array>
#include <charconv>
#include <iomanip>
#include <iostream>
#include <random>
#include <string_view>

#ifdef MC_PRICER_HYBRID
#include <omp.h>
#endif

namespace {

std::mt19937_64 make_generator(int rank, int thread_id) {
    const std::array<unsigned int, 3> seed_parts{
        42u, static_cast<unsigned int>(rank), static_cast<unsigned int>(thread_id)
    };
    // Thread zero uses the same stream as the pure MPI rank.
    const auto seed_end = seed_parts.begin() + (thread_id == 0 ? 2 : 3);
    std::seed_seq seed(seed_parts.begin(), seed_end);
    return std::mt19937_64(seed);
}

monte_carlo_detail::PayoffTotals simulate_local(const OptionParams& params, long path_count, int rank) {
    double sum = 0.0;
    double sum_sq = 0.0;

#ifdef MC_PRICER_HYBRID
    #pragma omp parallel reduction(+:sum, sum_sq)
    {
        const int thread_id = omp_get_thread_num();
#else
    {
        const int thread_id = 0;
#endif
        auto generator = make_generator(rank, thread_id);
        std::normal_distribution<double> normal(0.0, 1.0);

#ifdef MC_PRICER_HYBRID
        #pragma omp for schedule(static)
#endif
        for (long i = 0; i < path_count; ++i) {
            const double payoff = monte_carlo_detail::call_payoff(params, normal(generator));
            sum += payoff;
            sum_sq += payoff * payoff;
        }
    }

    return {path_count, sum, sum_sq};
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MC_PRICER_HYBRID
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "MPI_THREAD_FUNNELED support is required for the hybrid pricer.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
#else
    MPI_Init(&argc, &argv);
#endif

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long total_paths = 1000000L;
    if (rank == 0) {
        if (argc == 2) {
            const std::string_view argument(argv[1]);
            const char* end = argument.data() + argument.size();
            const auto parsed = std::from_chars(argument.data(), end, total_paths);
            if (parsed.ec != std::errc{} || parsed.ptr != end) {
                total_paths = 0;
            }
        } else if (argc > 2) {
            total_paths = 0;
        }
    }
    MPI_Bcast(&total_paths, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    if (total_paths < 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " [path_count >= 2]\n";
        }
        MPI_Finalize();
        return 1;
    }

    const OptionParams params{100.0, 100.0, 0.05, 0.2, 1.0};
    const long local_paths = total_paths / size + (rank < total_paths % size ? 1 : 0);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    const auto local = simulate_local(params, local_paths, rank);
    const double simulation_end = MPI_Wtime();

    monte_carlo_detail::PayoffTotals global;
    const std::array<double, 2> local_sums{local.sum, local.sum_sq};
    std::array<double, 2> global_sums{};
    MPI_Reduce(&local.path_count, &global.path_count, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_sums.data(), global_sums.data(), 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    const double reduction_end = MPI_Wtime();
    const std::array<double, 3> elapsed{
        reduction_end - start, simulation_end - start, reduction_end - simulation_end
    };
    std::array<double, 3> max_elapsed{};
    MPI_Reduce(elapsed.data(), max_elapsed.data(), 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        global.sum = global_sums[0];
        global.sum_sq = global_sums[1];
        const MCResult result = monte_carlo_detail::price_from_totals(params, global);
        std::cout << std::fixed << std::setprecision(4)
                  << "Black-Scholes analytical price: " << black_scholes_call_price(params) << '\n'
                  << "Ranks = " << size << "  N = " << global.path_count
                  << "  Price = " << result.price
                  << "  Standard Error = " << result.standard_error << '\n'
                  << "95% CI = [" << result.confidence_low << ", " << result.confidence_high << "]\n"
                  << std::setprecision(6)
                  << "Maximum rank time = " << max_elapsed[0] << " s\n"
                  << "Maximum local simulation time = " << max_elapsed[1] << " s\n"
                  << "Maximum payoff reduction time = " << max_elapsed[2] << " s\n";
    }

    MPI_Finalize();
    return 0;
}
