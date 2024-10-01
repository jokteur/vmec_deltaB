#pragma once

#include "types.h"

#include <vector>
#include <Kokkos_Sort.hpp>
#include "computation/array_math.h"
#include "computation/launch_kernel.h"

template <typename BinSort, typename Exec, typename... Arrays>
void sort_arrays(BinSort& bin_sort, Exec exec, size_t begin, size_t end, Arrays&... arrays) {
    // Sort each array using bin_sort
    (bin_sort.sort(exec, arrays, begin, end), ...);
}

/**
 * @brief Sorts the given arrays by the values in array_for_permutation
 *
 * @warning No values should be NaN or inf, please use replace_invalid_vals
 *
 * @note All arrays must have the same size, the same plateform, type, and must be 1D
 * @note The arrays are sorted in-place
 * @note The array_for_permutation is not modified unless it is in arrays_to_sort
 * @note The arrays are sorted in ascending order
 * @note Masks are ignored
 *
*/
// void sort_by(NdVectorView& array_for_permutation, std::vector<NdVectorView>& arrays_to_sort);
template<typename Arr, typename... Args>
void sort_by(const Arr& array_for_permutation, Args... arrays_to_sort) {
    static_assert(sizeof...(Args) > 0, "sort_by: at least one array to sort is required");
    static_assert((std::is_same_v<typename Arr::memory_space, typename Args::memory_space> && ...), "sort_by: all arrays must have the same plateform");

    using CompType = Kokkos::BinOp1D<Arr>;
    using Exec = typename Arr::execution_space;
    using range_policy = Kokkos::RangePolicy<Exec>;
    Kokkos::MinMaxScalar<typename Arr::non_const_value_type> result;
    Kokkos::MinMax<typename Arr::non_const_value_type> reducer(result);

    size_t begin = 0;
    size_t end = array_for_permutation.size();

    Kokkos::parallel_reduce("Kokkos::Sort::FindExtent", range_policy(Exec(), begin, end),
        Kokkos::Impl::min_max_functor<Arr>(array_for_permutation), reducer);
    if (result.min_val == result.max_val) return;
    Kokkos::BinSort<Arr, CompType> bin_sort(
        Exec(), array_for_permutation, begin, end,
        CompType((end - begin) / 2, result.min_val, result.max_val), true);

    bin_sort.create_permute_vector(Exec());
    (bin_sort.sort(Exec(), arrays_to_sort, begin, end), ...);
}

/**
 * @brief Replaces all NaN values and inf in the given array by the given value
 *
 * Use in sort
*/
template<typename Arr, typename T>
Arr replace_invalid_vals(const Arr& array, T value) {
    if constexpr (has_static_rank<Arr>()) {
        static_assert(Arr::rank() == 1, "replace_invalid_vals only supports 1D arrays");
    }
    else {
        if (array.rank() != 1) {
            throw std::runtime_error("replace_invalid_vals only supports 1D arrays");
        }
    }

    using Space = typename Arr::memory_space;
    using Exec = typename Arr::execution_space;
    using value_t = typename Arr::value_type;

    Arr output("output", array.extent(0));
    launch<Space>("replace_invalid_vals", array.extent(0), KOKKOS_LAMBDA(LAMBDA_ARGS) {
        if (Kokkos::isnan(array(i)) || Kokkos::isinf(array(i))) {
            output(i) = value;
        }
        else {
            output(i) = array(i);
        }
    });
    return output;
}