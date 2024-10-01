#pragma once

#include "local_array.h"

/**
 * @brief Generate an NdArray with values from 0 to N0 * N1 * N2
 * 
 * @tparam T Data type
 * @tparam N0 Size of the first dimension
 * @tparam N1 Size of the second dimension
 * @tparam N2 Size of the third dimension
 * 
 * If N0 > 1, N1 == 1 and N2 == 1, the output will be a 1D array
 * If N0, N1 > 1 and N2 == 1, the output will be a 2D array
 * If N0, N1, N2 > 1, the output will be a 3D array
 * 
 * @return NdArray<T, N0, N1, N2> NdArray with values from 0 to N0 * N1 * N2
 * 
 * @example auto a = arange<double, 2, 2>(); // a = [[0, 1], [2, 3]]
 * @example auto b = arange<double, 2, 2, 2>(); // b = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
*/
template<typename T, size_t N0, size_t N1 = 1, size_t N2 = 1>
KOKKOS_INLINE_FUNCTION decltype(auto) arange() {
    using ArrType = NdArray<T, N0, N1, N2>;

    NdArray<T, N0, N1, N2> out;
    for (size_t i = 0;i < N0;i++) {
        for (size_t j = 0;j < N1;j++) {
            for (size_t k = 0;k < N2;k++) {
                if constexpr (NdArray<T, N0, N1, N2>::rank() == 3) {
                    out(i, j, k) = i * N1 * N2 + j * N2 + k;
                }
                else if constexpr (NdArray<T, N0, N1, N2>::rank() == 2) {
                    out(i, j) = i * N1 + j;
                }
                else if constexpr (NdArray<T, N0, N1, N2>::rank() == 1) {
                    out(i) = i;
                }
            }
        }
    }
    return out;
}

