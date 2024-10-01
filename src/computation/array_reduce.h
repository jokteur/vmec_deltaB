#pragma once
#include <mpi.h>
#include "io/output/print.h"
#include "util/template_util.h"
#include "mask.h"
#include <limits>

/**
 * Sums 1D array (by default local)
 * 
 * @param arr Array to sum
 * @param mask Mask to apply to array
 * @param local_only If true, only sum locally (on the MPI process). Beware that setting it to false calls an MPI_Barrier. Make sure that all processes are calling this.
*/
template<typename Arr, typename T = typename Arr::value_type, typename Space = typename Arr::memory_space>
T sum(const Arr& arr, const Mask<Space>& mask = Mask<Space>(), bool local_only = true) {
    if constexpr (has_static_rank<Arr>()) {
        static_assert(Arr::rank == 1, "sum (reduce) only supports 1D arrays");
    }
    else {
        if (arr.rank() != 1) {
            throw std::runtime_error("sum (reduce) only supports 1D arrays");
        }
    }

    using Exec = typename Arr::execution_space;

    auto policy = Kokkos::RangePolicy<Exec>(0, arr.extent(0));
    T sum = 0.0;
    if (!mask.is_wildcard()) {
        Kokkos::parallel_reduce("reduce sum", policy, KOKKOS_LAMBDA(const size_t & i, T & lsum) {
            if (mask(i))
                lsum += arr(i);
        }, sum);
    }
    else {
        Kokkos::parallel_reduce("reduce sum", policy, KOKKOS_LAMBDA(const size_t & i, T & lsum) {
            lsum += arr(i);
        }, sum);
    }

    if (local_only) {
        return sum;
    }
    else {
        double global_sum = 0.0;
        MPI_Allreduce(&sum, &global_sum, 1, get_mpi_type<T>(), MPI_SUM, MPI_COMM_WORLD);
        return global_sum;
    }
}

/**
 * @brief Product of 1D array (by default local)
 * 
 * @param arr Array to multiply
 * @param mask Mask to apply to array
 * @param local_only If true, only multiply locally (on the MPI process). Beware that setting it to false calls an MPI_Barrier. Make sure that all processes are calling this.
*/
template<typename Arr, typename T = typename Arr::value_type, typename Space = typename Arr::memory_space>
T prod(const Arr& arr, const Mask<Space>& mask = Mask<Space>(), bool local_only = true) {
    if constexpr (has_static_rank<Arr>()) {
        static_assert(Arr::rank == 1, "prod (reduce) only supports 1D arrays");
    }
    else {
        if (arr.rank() != 1) {
            throw std::runtime_error("prod (reduce) only supports 1D arrays");
        }
    }

    using Exec = typename Arr::execution_space;

    auto policy = Kokkos::RangePolicy<Exec>(0, arr.extent(0));
    T prod = 1.0;
    if (!mask.is_wildcard()) {
        Kokkos::parallel_reduce("reduce prod", policy, KOKKOS_LAMBDA(const size_t & i, T & lprod) {
            if (mask(i))
                lprod *= arr(i);
        }, prod);
    }
    else {
        Kokkos::parallel_reduce("reduce prod", policy, KOKKOS_LAMBDA(const size_t & i, T & lprod) {
            lprod *= arr(i);
        }, prod);
    }

    if (local_only) {
        return prod;
    }
    else {
        double global_prod = 0.0;
        MPI_Allreduce(&prod, &global_prod, 1, get_mpi_type<T>(), MPI_PROD, MPI_COMM_WORLD);
        return global_prod;
    }
}

/**
 * @brief Minimum of 1D array (by default local)
 * 
 * @param arr Array to find minimum
 * @param mask Mask to apply to array
 * @param local_only If true, only find minimum locally (on the MPI process). Beware that setting it to false calls an MPI_Barrier. Make sure that all processes are calling this.
*/
template<typename Arr, typename T = typename Arr::value_type, typename Space = typename Arr::memory_space>
T min(const Arr& arr, const Mask<Space>& mask = Mask<Space>(), bool local_only = true) {
    if constexpr (has_static_rank<Arr>()) {
        static_assert(Arr::rank == 1, "min (reduce) only supports 1D arrays");
    }
    else {
        if (arr.rank() != 1) {
            throw std::runtime_error("min (reduce) only supports 1D arrays");
        }
    }

    using Exec = typename Arr::execution_space;

    auto policy = Kokkos::RangePolicy<Exec>(0, arr.extent(0));
    T min = std::numeric_limits<T>::max();
    if (!mask.is_wildcard()) {
        Kokkos::parallel_reduce("reduce min", policy, KOKKOS_LAMBDA(const size_t & i, T & lmin) {
            if (mask(i))
                lmin = lmin < arr(i) ? lmin : arr(i);
        }, Kokkos::Min<T>(min));
    }
    else {
        Kokkos::parallel_reduce("reduce min", policy, KOKKOS_LAMBDA(const size_t & i, T & lmin) {
            lmin = lmin < arr(i) ? lmin : arr(i);
        }, Kokkos::Min<T>(min));
    }

    if (local_only) {
        return min;
    }
    else {
        T global_min;
        MPI_Allreduce(&min, &global_min, 1, get_mpi_type<T>(), MPI_MIN, MPI_COMM_WORLD);
        return global_min;
    }
}

/**
 * @brief Maximum of 1D array (by default local)
 * 
 * @param arr Array to find maximum
 * @param mask Mask to apply to array
 * @param local_only If true, only find maximum locally (on the MPI process). Beware that setting it to false calls an MPI_Barrier. Make sure that all processes are calling this.
*/
template<typename Arr, typename T = typename Arr::value_type, typename Space = typename Arr::memory_space>
T max(const Arr& arr, const Mask<Space>& mask = Mask<Space>(), bool local_only = true) {
    if constexpr (has_static_rank<Arr>()) {
        static_assert(Arr::rank == 1, "max (reduce) only supports 1D arrays");
    }
    else {
        if (arr.rank() != 1) {
            throw std::runtime_error("max (reduce) only supports 1D arrays");
        }
    }

    using Exec = typename Arr::execution_space;

    auto policy = Kokkos::RangePolicy<Exec>(0, arr.extent(0));
    T max = std::numeric_limits<T>::min();
    if (!mask.is_wildcard()) {
        Kokkos::parallel_reduce("reduce max", policy, KOKKOS_LAMBDA(const size_t & i, T & lmax) {
            if (mask(i))
                lmax = lmax > arr(i) ? lmax : arr(i);
        }, Kokkos::Max<T>(max));
    }
    else {
        Kokkos::parallel_reduce("reduce max", policy, KOKKOS_LAMBDA(const size_t & i, T & lmax) {
            lmax = lmax > arr(i) ? lmax : arr(i);
        }, Kokkos::Max<T>(max));
    }

    if (local_only) {
        return max;
    }
    else {
        T global_max;
        MPI_Allreduce(&max, &global_max, 1, get_mpi_type<T>(), MPI_MAX, MPI_COMM_WORLD);
        return global_max;
    }
}

/**
 * @brief Sums a Kokkos array across all MPI ranks in place
 *
 * As this is using MPI function, the array is converted to CPU first
*/
template<typename Arr>
void inplace_sum(Arr& arr) {
    using T = typename Arr::value_type;
    using data_type = typename Arr::data_type;
    using Space = typename Arr::memory_space;
    using Exec = typename Arr::execution_space;

    if constexpr (std::is_same_v<Space, HOST>) {
        MPI_Allreduce(MPI_IN_PLACE, arr.data(), arr.size(), get_mpi_type<T>(), MPI_SUM, MPI_COMM_WORLD);
    }
    else {
        ///@todo Use GPU aware MPI ?
        auto arr_cpu = to_cpu<Kokkos::View<data_type, HOST>>(arr);
        inplace_sum(arr_cpu);
    }

}