#pragma once

#include "types.h"
#include "util/template_util.h"
#include <limits>
#include <exception>
#include <vector>

template<int Rank, typename T, typename = typename std::enable_if<is_kokkos_view<T>()>::type>
decltype(auto) to_fixed_sized_rank(const T& input) {
    using Space = typename T::memory_space;
    using Type = typename T::value_type;

    if constexpr (Rank == 1) {
        Kokkos::View<Type*, Space> output(input.label(), input.extent(0));
        Kokkos::deep_copy(output, input);
        return output;
    }
    else if constexpr (Rank == 2) {
        Kokkos::View<Type**, Space> output(input.label(), input.extent(0), input.extent(1));
        Kokkos::deep_copy(output, input);
        return output;
    }
    else if constexpr (Rank == 3) {
        Kokkos::View<Type***, Space> output(input.label(), input.extent(0), input.extent(1), input.extent(2));
        Kokkos::deep_copy(output, input);
        return output;
    }
    else if constexpr (Rank == 4) {
        Kokkos::View<Type****, Space> output(input.label(), input.extent(0), input.extent(1), input.extent(2), input.extent(3));
        Kokkos::deep_copy(output, input);
        return output;
    }
    else if constexpr (Rank == 5) {
        Kokkos::View<Type*****, Space> output(input.label(), input.extent(0), input.extent(1), input.extent(2), input.extent(3), input.extent(4));
        Kokkos::deep_copy(output, input);
        return output;
    }
    else if constexpr (Rank == 6) {
        Kokkos::View<Type******, Space> output(input.label(), input.extent(0), input.extent(1), input.extent(2), input.extent(3), input.extent(4), input.extent(5));
        Kokkos::deep_copy(output, input);
        return output;
    }
    else {
        Kokkos::View<Type*******, Space> output(input.label(), input.extent(0), input.extent(1), input.extent(2), input.extent(3), input.extent(4), input.extent(5), input.extent(6));
        Kokkos::deep_copy(output, input);
        return output;
    }
    static_assert(Rank > 0 && Rank < 8, "Rank must be between 1 and 7");
}

template<typename OutType, typename T, typename = typename std::enable_if<is_kokkos_view<T>()>::type>
OutType to_cpu(const T& input) {
    auto mirror = Kokkos::create_mirror_view(input);
    if constexpr (OutType::Rank == 1) {
        OutType output(input.label(), input.extent(0));
        // This is doing two copies, but we want to force to OutType
        Kokkos::deep_copy(mirror, input);
        Kokkos::deep_copy(output, mirror);
        return output;
    }
    else if constexpr (OutType::Rank == 2) {
        OutType output(input.label(), input.extent(0), input.extent(1));
        Kokkos::deep_copy(mirror, input);
        Kokkos::deep_copy(output, mirror);
        return output;
    }
    else if constexpr (OutType::Rank == 3) {
        OutType output(input.label(), input.extent(0), input.extent(1), input.extent(2));
        Kokkos::deep_copy(mirror, input);
        Kokkos::deep_copy(output, mirror);
        return output;
    }
    else if constexpr (OutType::Rank == 4) {
        OutType output(input.label(), input.extent(0), input.extent(1), input.extent(2), input.extent(3));
        Kokkos::deep_copy(mirror, input);
        Kokkos::deep_copy(output, mirror);
        return output;
    }
    else if constexpr (OutType::Rank == 5) {
        OutType output(input.label(), input.extent(0), input.extent(1), input.extent(2), input.extent(3), input.extent(4));
        Kokkos::deep_copy(mirror, input);
        Kokkos::deep_copy(output, mirror);
        return output;
    }
    else if constexpr (OutType::Rank == 6) {
        OutType output(input.label(), input.extent(0), input.extent(1), input.extent(2), input.extent(3), input.extent(4), input.extent(5));
        Kokkos::deep_copy(mirror, input);
        Kokkos::deep_copy(output, mirror);
        return output;
    }
    else {
        OutType output(input.label(), input.extent(0), input.extent(1), input.extent(2), input.extent(3), input.extent(4), input.extent(5), input.extent(6));
        Kokkos::deep_copy(mirror, input);
        Kokkos::deep_copy(output, mirror);
        return output;
    }
}

template<typename OutType, typename T, typename = typename std::enable_if<is_kokkos_view<T>()>::type>
OutType to_gpu(const T& input) {
    if constexpr (T::Rank == 1) {
        OutType output(input.label(), input.extent(0));
        auto host_arr = Kokkos::create_mirror_view(output);
        Kokkos::deep_copy(host_arr, input);
        Kokkos::deep_copy(output, host_arr);
        return output;
    }
    else if constexpr (T::Rank == 2) {
        OutType output(input.label(), input.extent(0), input.extent(1));
        auto host_arr = Kokkos::create_mirror_view(output);
        Kokkos::deep_copy(host_arr, input);
        Kokkos::deep_copy(output, host_arr);
        return output;
    }
    else if constexpr (T::Rank == 3) {
        OutType output(input.label(), input.extent(0), input.extent(1), input.extent(2));
        auto host_arr = Kokkos::create_mirror_view(output);
        Kokkos::deep_copy(host_arr, input);
        Kokkos::deep_copy(output, host_arr);
        return output;
    }
    else if constexpr (T::Rank == 4) {
        OutType output(input.label(), input.extent(0), input.extent(1), input.extent(2), input.extent(3));
        auto host_arr = Kokkos::create_mirror_view(output);
        Kokkos::deep_copy(host_arr, input);
        Kokkos::deep_copy(output, host_arr);
        return output;
    }
    else if constexpr (T::Rank == 5) {
        OutType output(input.label(), input.extent(0), input.extent(1), input.extent(2), input.extent(3), input.extent(4));
        auto host_arr = Kokkos::create_mirror_view(output);
        Kokkos::deep_copy(host_arr, input);
        Kokkos::deep_copy(output, host_arr);
        return output;
    }
    else if constexpr (T::Rank == 6) {
        OutType output(input.label(), input.extent(0), input.extent(1), input.extent(2), input.extent(3), input.extent(4), input.extent(5));
        auto host_arr = Kokkos::create_mirror_view(output);
        Kokkos::deep_copy(host_arr, input);
        Kokkos::deep_copy(output, host_arr);
        return output;
    }
    else {
        OutType output(input.label(), input.extent(0), input.extent(1), input.extent(2), input.extent(3), input.extent(4), input.extent(5), input.extent(6));
        auto host_arr = Kokkos::create_mirror_view(output);
        Kokkos::deep_copy(host_arr, input);
        Kokkos::deep_copy(output, host_arr);
        return output;
    }
}

template<typename T, typename Space, typename U>
Kokkos::View<T*, Space> vector_to_kokkos_view(std::vector<U>& input) {
    Kokkos::View<T*, HOST> output("", input.size());
    auto policy = Kokkos::RangePolicy<HOSTexec>(0, input.size());
    Kokkos::parallel_for("vector_to_kokkos_view", policy, [=](const size_t& i) {
        output(i) = input[i];
    });
    if constexpr (std::is_same_v<Space, HOST>) {
        return output;
    }
    else {
        return to_gpu<Kokkos::View<T*, Space>>(output);
    }
}

/**
 * @brief Returns a vector of evenly spaced numbers over a specified interval.
*/
template<typename View>
View linspace(double start, double end, int num, bool endpoint = true) {
    View result("linspace", num);
    double step = (end - start) / (num - (endpoint ? 1 : 0));

    auto range_policy = Kokkos::RangePolicy<typename View::execution_space>(0, num);
    Kokkos::parallel_for("Make linspace", range_policy, KOKKOS_LAMBDA(const size_t & i) {
        result(i) = start + i * step;
    });
    return result;
}