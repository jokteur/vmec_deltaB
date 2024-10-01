/** @file */
#pragma once

#include <type_traits>
#include "types.h"
#include "set_architecture.h"
#include "mask.h"

inline size_t find_next_multiple_of(size_t n, size_t m) {
    return n + m - n % m;
}

template<typename Space>
inline size_t get_team_size() {
    if constexpr (std::is_same_v<Space, HOST>) {
        return Kokkos::num_threads();
    }
    else {
        return 128;
    }
}

template<typename Space>
struct TeamPolicy {
    using exec = typename Kokkos::View<ARCH_TYPE*, Space>::execution_space;
    using type = Kokkos::TeamPolicy<exec>;
};
template<typename Space>
struct MemberType {
    using exec = typename Kokkos::View<ARCH_TYPE*, Space>::execution_space;
    using type = typename Kokkos::TeamPolicy<exec>::member_type;
};

template<typename Space>
struct ScratchType {
    using exec = typename Kokkos::View<ARCH_TYPE*, Space>::execution_space;
    using type = Kokkos::View<ARCH_TYPE*, typename exec::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
};

using member_type_t = typename MemberType<ARCH>::type;
using member_type_cpu_t = typename MemberType<HOST>::type;

#define LAMBDA_ARGS const size_t& i
#define LAMBDA_ARGS_HOST const size_t& i
#define LAMBDA_NOCPY(...) [__VA_ARGS__] __host__ __device__ (LAMBDA_ARGS)
#define LAMBDA_NOCPY_HOST(...) [__VA_ARGS__] __host__ __device__ (LAMBDA_ARGS_HOST)


template<typename Space>
decltype(auto) get_policy(size_t num_particles) {
    size_t team_size = get_team_size<Space>();
    using t_team_policy = typename TeamPolicy<Space>::type;
    return t_team_policy(num_particles / team_size + 1, team_size);
}

/**
 * @brief launch a 1D kokkos kernel on the specified architecture
 *
 * @tparam Space on which space the kernel is launched (CPU or GPU)
 * @tparam Func type of the functor
 * @param name of the kernel ; please use clear unique names, such that profiling is easier
 * @param num_particles size of the kernel
 * @param f lambda or functor
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
template<typename Space, class Func>
void launch(const std::string& name, size_t num_particles, Func&& f) {
    using t_team_policy = typename TeamPolicy<Space>::type;
    using t_member_type = typename MemberType<Space>::type;
    using Exec = typename Kokkos::View<double*, Space>::execution_space;

    // auto team_policy = get_policy<Space>(num_particles); //.set_scratch_size(0, Kokkos::PerThread(46 * sizeof(double)));
    auto team_policy = Kokkos::RangePolicy<Exec>(0, num_particles);
    if constexpr (std::is_same_v<Space, HOST>) {
        Kokkos::parallel_for(name, team_policy, [=](const size_t& i) {
            // size_t i = team_member.league_rank() * team_member.team_size() + team_member.team_rank();
            // if (i < num_particles)
            f(i);
            });
    }
    else {
        Kokkos::parallel_for(name, team_policy, KOKKOS_LAMBDA(const size_t & i) {
            // size_t i = team_member.league_rank() * team_member.team_size() + team_member.team_rank();
            // if (i < num_particles)
            f(i);
        });
    }
}

/**
 * @brief Launch a 1D kokkos kernel with mask
 *
 * @tparam Space on which space the kernel is launched (CPU or GPU)
 * @tparam Func type of the functor
 * @param name of the kernel ; please use clear unique names, such that profiling is easier
 * @param num_particles size of the kernel
 * @param mask mask of true/false
 * @param f functor or lambda function
 */
template<typename Space, class Func>
void launch(const std::string& name, size_t num_particles, const Mask<Space>& mask, Func&& f) {
    using t_team_policy = typename TeamPolicy<Space>::type;
    using t_member_type = typename MemberType<Space>::type;
    using Exec = typename Kokkos::View<double*, Space>::execution_space;

    // auto team_policy = get_policy<Space>(num_particles); //.set_scratch_size(0, Kokkos::PerThread(46 * sizeof(double)));
    auto team_policy = Kokkos::RangePolicy<Exec>(0, num_particles);
    if (mask.is_wildcard()) {
        launch<Space>(name, num_particles, f);
    }
    else {
        if constexpr (std::is_same_v<Space, HOST>) {
            Kokkos::parallel_for(name, team_policy, [=](const size_t& i) {
                // size_t i = team_member.league_rank() * team_member.team_size() + team_member.team_rank();
                if (mask(i))
                    f(i);
                });
        }
        else {
            Kokkos::parallel_for(name, team_policy, KOKKOS_LAMBDA(const size_t & i) {
                // size_t i = team_member.league_rank() * team_member.team_size() + team_member.team_rank();
                if (mask(i))
                    f(i);
            });
        }
    }
}
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop