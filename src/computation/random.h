#pragma once

#include "types.h"
#include <initializer_list>
#include <random>
#include <chrono>
#include <thread>
#include <mpi.h>
#include "util/template_util.h"

// we only use the address of this function
static void seed_function() {}

template<int Rank, typename View, typename... Args>
View rand(std::initializer_list<size_t> sizes, double low, double high, const std::string& label = "random", Args... seed) {
    std::vector<size_t> sizes_v(sizes);

    constexpr size_t rank = std::integral_constant<size_t, Rank>::value;

    Kokkos::Random_XorShift64_Pool<typename View::execution_space> random_pool;
    if constexpr (sizeof...(seed) > 0) {
        random_pool = Kokkos::Random_XorShift64_Pool<typename View::execution_space>(get<0>(seed...));
    }
    else {
        // Seed generation (time, CPU based)
        // random_device may be deterministic, so we need to generate a seed based on
        // "random" events, like current time, CPU memory layout, ...
        // Based on https://stackoverflow.com/a/34490647
        static long long seed_counter = 0;
        int var;
        void* x = std::malloc(sizeof(int));
        free(x);
        std::seed_seq seed{
            // Time
            static_cast<long long>(std::chrono::high_resolution_clock::now()
                                       .time_since_epoch()
                                       .count()),
            // ASLR
            static_cast<long long>(reinterpret_cast<intptr_t>(&seed_counter)),
            static_cast<long long>(reinterpret_cast<intptr_t>(&var)),
            static_cast<long long>(reinterpret_cast<intptr_t>(x)),
            static_cast<long long>(reinterpret_cast<intptr_t>(&seed_function)),
            static_cast<long long>(reinterpret_cast<intptr_t>(&_Exit)),
            // Thread id
            static_cast<long long>(
                std::hash<std::thread::id>()(std::this_thread::get_id())),
            // counter
            ++seed_counter };
        std::default_random_engine eng(seed);
        constexpr static uint64_t MAX_URAND64 = std::numeric_limits<uint64_t>::max();
        std::uniform_int_distribution<size_t> uniform_dist(0, MAX_URAND64);
        size_t seed_i = uniform_dist(eng);
        random_pool = Kokkos::Random_XorShift64_Pool<>(seed_i);
    }

    if constexpr (rank == 1) {
        View result(label, sizes_v[0]);
        auto range_policy = Kokkos::RangePolicy<typename View::execution_space>(0, sizes_v[0]);
        Kokkos::parallel_for("Fill random 1D", range_policy, KOKKOS_LAMBDA(const size_t & i) {
            auto generator = random_pool.get_state();
            result(i) = generator.drand(low, high);
            random_pool.free_state(generator);
        });
        return result;
    }
    else if constexpr (rank == 2) {
        auto range_policy = Kokkos::MDRangePolicy<typename View::execution_space, Kokkos::Rank<2>>({ 0, 0 }, { sizes_v[0], sizes_v[1] });
        View result(label, sizes_v[0], sizes_v[1]);
        Kokkos::parallel_for("Fill random 2D", range_policy, KOKKOS_LAMBDA(const size_t & i, const size_t & j) {
            auto generator = random_pool.get_state();
            result(i, j) = generator.drand(low, high);
            random_pool.free_state(generator);
        });
        return result;
    }
    else if constexpr (rank == 3) {
        auto range_policy = Kokkos::MDRangePolicy<typename View::execution_space, Kokkos::Rank<3>>({ 0, 0, 0 }, { sizes_v[0], sizes_v[1], sizes_v[2] });
        View result(label, sizes_v[0], sizes_v[1], sizes_v[2]);
        Kokkos::parallel_for("Fill random 3D", range_policy, KOKKOS_LAMBDA(const size_t & i, const size_t & j, const size_t & k) {
            auto generator = random_pool.get_state();
            result(i, j, k) = generator.drand(low, high);
            random_pool.free_state(generator);
        });
        return result;
    }
    else if constexpr (rank == 4) {
        auto range_policy = Kokkos::MDRangePolicy<typename View::execution_space, Kokkos::Rank<4>>({ 0, 0, 0, 0 }, { sizes_v[0], sizes_v[1], sizes_v[2], sizes_v[3] });
        View result(label, sizes_v[0], sizes_v[1], sizes_v[2], sizes_v[3]);
        Kokkos::parallel_for("Fill random 4D", range_policy, KOKKOS_LAMBDA(const size_t & i, const size_t & j, const size_t & k, const size_t & l) {
            auto generator = random_pool.get_state();
            result(i, j, k, l) = generator.drand(low, high);
            random_pool.free_state(generator);
        });
        return result;
    }
    else {
        throw std::runtime_error("Random array generation only supported for 1D, 2D, 3D and 4D arrays");
    }
}


/**
 * @brief Retrieve a random seed for the random number generator
 * 
 * Call this function everytime you need a seed (increments a global counter, which take into account
 * the MPI rank of the process).
 * 
 * @param init_seed The initial seed to use.
*/
size_t random_seed(size_t init_seed = SIZE_MAX);