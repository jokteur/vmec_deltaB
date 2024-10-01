#pragma once
#include "array_reduce.h"
#include "array_math.h"
#include "functions.h"
#include "mask.h"
#include "io/output/print.h"

template<typename Arr, typename T = typename Arr::value_type, typename Space = typename Arr::memory_space>
T mean(const Arr& arr, const Mask<Space>& mask = Mask<Space>(), bool local_only = true) {
    if (mask.is_wildcard()) {
        return sum(arr, Mask<Space>(), local_only) / arr.size();
    }
    else {
        return sum(arr, mask, local_only) / mask.size(arr.size());
    }
}

template<typename Arr, typename T = typename Arr::value_type, typename Space = typename Arr::memory_space>
T variance(T mean, const Arr& arr, const Mask<Space>& mask = Mask<Space>(), bool local_only = true) {
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
    T var = 0.0;
    if (!mask.is_wildcard()) {
        Kokkos::parallel_reduce("reduce sum", policy, KOKKOS_LAMBDA(const size_t & i, T & lsum) {
            if (mask(i))
                lsum += pow2(arr(i) - mean);
        }, var);
        int size = mask.size(arr.size());

        if (local_only) {
            return var / size;
        }
        else {
            double global_variance = 0.0;
            MPI_Reduce(&var, &global_variance, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            int global_sum = 0.0;
            MPI_Reduce(&size, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            return global_variance / global_sum;
        }
    }
    else {
        Kokkos::parallel_reduce("reduce sum", policy, KOKKOS_LAMBDA(const size_t & i, T & lsum) {
            lsum += pow2(arr(i) - mean);
        }, var);
        int size = arr.size();
        if (local_only) {
            return var / size;
        }
        else {
            double global_variance = 0.0;
            MPI_Reduce(&var, &global_variance, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            int global_sum = 0.0;
            MPI_Reduce(&size, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            return global_variance / global_sum;
        }
    }
}

template<typename Arr, typename T = typename Arr::value_type, typename Space = typename Arr::memory_space>
T variance(const Arr& arr, const Mask<Space>& mask = Mask<Space>(), bool local_only = true) {
    return variance(mean(arr, mask, local_only), arr, mask, local_only);
}

template<typename Arr, typename T = typename Arr::value_type, typename Space = typename Arr::memory_space>
T std_dev(T mean, const Arr& arr, const Mask<Space>& mask = Mask<Space>(), bool local_only = true) {
    return Kokkos::sqrt(variance(mean, arr, mask, local_only));
}

template<typename Arr, typename T = typename Arr::value_type, typename Space = typename Arr::memory_space>
T std_dev(const Arr& arr, const Mask<Space>& mask = Mask<Space>(), bool local_only = true) {
    return Kokkos::sqrt(variance(mean(arr, mask, local_only), arr, mask, local_only));
}