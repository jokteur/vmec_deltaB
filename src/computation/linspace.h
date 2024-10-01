#pragma once

#include "types.h"

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