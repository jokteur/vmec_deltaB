#pragma once
#include "types.h"

template<typename T>
KOKKOS_INLINE_FUNCTION T pow2(const T& x) {
    return x * x;
}
template<typename T>
KOKKOS_INLINE_FUNCTION T pow3(const T& x) {
    return x * x * x;
}
template<typename T>
KOKKOS_INLINE_FUNCTION T pow4(const T& x) {
    return x * x * x * x;
}

#define PI 3.14159265358979323846

template<typename T>
KOKKOS_INLINE_FUNCTION T normalise_angle(const T& angle) {
    return Kokkos::fmod(Kokkos::fmod(angle, 2 * PI) + 2 * PI, 2 * PI);
}

#undef PI

template<typename T>
KOKKOS_INLINE_FUNCTION T sign(const T& x) {
    return double(x > 0) - double(x < 0);
}
