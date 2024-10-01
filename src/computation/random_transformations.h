#pragma once
#include "random.h"
#include "functions.h"

template<typename T>
KOKKOS_INLINE_FUNCTION T gaussian(T u1, T u2) {
    // Box-Muller transformation, for mu = 0, sigma = 1
    return Kokkos::sqrt(-2.0f * Kokkos::log(u1)) * Kokkos::sin(2.0f * pi * u2);
}

template<int Rank, typename View, typename... Args>
View gaussian(std::initializer_list<size_t> sizes, Args... seed) {
    // for mu = 0, sigma = 1    
    std::string label = "gaussian";
    std::vector<size_t> sizes_v(sizes);
    constexpr size_t rank = std::integral_constant<size_t, Rank>::value;

    auto u1 = rand<Rank, View>(sizes, 0.0, 1.0, "rand", seed...);
    auto u2 = rand<Rank, View>(sizes, 0.0, 1.0, "rand", seed...);
    if constexpr (rank == 1) {
        View result(label, sizes_v[0]);
        auto range_policy = Kokkos::RangePolicy<typename View::execution_space>(0, sizes_v[0]);
        Kokkos::parallel_for("Fill random 1D", range_policy, KOKKOS_LAMBDA(const size_t & i) {
            result(i) = gaussian(u1(i), u2(i));
        });
        return result;
    }
    else if constexpr (rank == 2) {
        auto range_policy = Kokkos::MDRangePolicy<typename View::execution_space, Kokkos::Rank<2>>({ 0, 0 }, { sizes_v[0], sizes_v[1] });
        View result(label, sizes_v[0], sizes_v[1]);
        Kokkos::parallel_for("Fill random 2D", range_policy, KOKKOS_LAMBDA(const size_t & i, const size_t & j) {
            result(i, j) = gaussian(u1(i, j), u2(i, j));
        });
        return result;
    }
    else if constexpr (rank == 3) {
        auto range_policy = Kokkos::MDRangePolicy<typename View::execution_space, Kokkos::Rank<3>>({ 0, 0, 0 }, { sizes_v[0], sizes_v[1], sizes_v[2] });
        View result(label, sizes_v[0], sizes_v[1], sizes_v[2]);
        Kokkos::parallel_for("Fill random 3D", range_policy, KOKKOS_LAMBDA(const size_t & i, const size_t & j, const size_t & k) {
            result(i, j, k) = gaussian(u1(i, j, k), u2(i, j, k));
        });
        return result;
    }
    else if constexpr (rank == 4) {
        auto range_policy = Kokkos::MDRangePolicy<typename View::execution_space, Kokkos::Rank<4>>({ 0, 0, 0, 0 }, { sizes_v[0], sizes_v[1], sizes_v[2], sizes_v[3] });
        View result(label, sizes_v[0], sizes_v[1], sizes_v[2], sizes_v[3]);
        Kokkos::parallel_for("Fill random 4D", range_policy, KOKKOS_LAMBDA(const size_t & i, const size_t & j, const size_t & k, const size_t & l) {
            result(i, j, k, l) = gaussian(u1(i, j, k, l), u2(i, j, k, l));
        });
        return result;
    }
    else {
        throw std::runtime_error("Random array generation only supported for 1D, 2D, 3D and 4D arrays");
    }
}

template<int Rank, typename View, typename... Args>
View plusminus(std::initializer_list<size_t> sizes, Args... seed) {
    auto pm = rand<Rank, View>(sizes, -1.0, 1.0, "rand", seed...);
    std::string label = "plusminus";
    std::vector<size_t> sizes_v(sizes);
    constexpr size_t rank = std::integral_constant<size_t, Rank>::value;

    if constexpr (rank == 1) {
        View result(label, sizes_v[0]);
        auto range_policy = Kokkos::RangePolicy<typename View::execution_space>(0, sizes_v[0]);
        Kokkos::parallel_for("Fill random 1D", range_policy, KOKKOS_LAMBDA(const size_t & i) {
            result(i) = sign(pm(i));
        });
        return result;
    }
    else if constexpr (rank == 2) {
        auto range_policy = Kokkos::MDRangePolicy<typename View::execution_space, Kokkos::Rank<2>>({ 0, 0 }, { sizes_v[0], sizes_v[1] });
        View result(label, sizes_v[0], sizes_v[1]);
        Kokkos::parallel_for("Fill random 2D", range_policy, KOKKOS_LAMBDA(const size_t & i, const size_t & j) {
            result(i, j) = sign(pm(i, j));
        });
        return result;
    }
    else if constexpr (rank == 3) {
        auto range_policy = Kokkos::MDRangePolicy<typename View::execution_space, Kokkos::Rank<3>>({ 0, 0, 0 }, { sizes_v[0], sizes_v[1], sizes_v[2] });
        View result(label, sizes_v[0], sizes_v[1], sizes_v[2]);
        Kokkos::parallel_for("Fill random 3D", range_policy, KOKKOS_LAMBDA(const size_t & i, const size_t & j, const size_t & k) {
            result(i, j, k) = sign(pm(i, j, k));
        });
        return result;
    }
    else if constexpr (rank == 4) {
        auto range_policy = Kokkos::MDRangePolicy<typename View::execution_space, Kokkos::Rank<4>>({ 0, 0, 0, 0 }, { sizes_v[0], sizes_v[1], sizes_v[2], sizes_v[3] });
        View result(label, sizes_v[0], sizes_v[1], sizes_v[2], sizes_v[3]);
        Kokkos::parallel_for("Fill random 4D", range_policy, KOKKOS_LAMBDA(const size_t & i, const size_t & j, const size_t & k, const size_t & l) {
            result(i, j, k, l) = sign(pm(i, j, k, l));
        });
        return result;
    }
    else {
        throw std::runtime_error("Random array generation only supported for 1D, 2D, 3D and 4D arrays");
    }
}