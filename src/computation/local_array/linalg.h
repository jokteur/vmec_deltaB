#pragma once

#include "local_array.h"

#define TEMPLATE_DECL template<typename T, typename U,                             \
    typename = typename std::enable_if<is_ndarray<T>() || is_subview<T>()>::type,  \
    typename = typename std::enable_if<is_ndarray<U>() || is_subview<U>()>::type   \
>

/**
 * @brief Dot product of two effective vectors
 * 
 * @tparam T Type of the first vector (ndarray or subview)
 * @tparam U Type of the second vector (ndarray or subview)
 * @param lhs First vector
 * @param rhs Second vector
 * @return auto Dot product of the two vectors
*/
TEMPLATE_DECL
KOKKOS_INLINE_FUNCTION decltype(auto) dot(const T& lhs, const U& rhs) {
    if constexpr (is_ndarray<T>() && is_ndarray<U>()) {
        return dot(lhs.as_subview_const(), rhs.as_subview_const());
    }
    else if constexpr (is_ndarray<T>() && is_subview<U>()) {
        return dot(lhs.as_subview_const(), rhs.const_ref());
    }
    else if constexpr (is_subview<T>() && is_ndarray<U>()) {
        return dot(lhs.const_ref(), rhs.as_subview_const());
    }
    else {
        static_assert(T::effective_rank() == 1 && U::effective_rank() == 1, "Must have two vectors to perform dot product");
        constexpr auto lhs_shape = T::effective_shape();
        constexpr auto rhs_shape = U::effective_shape();
        static_assert(get<0>(lhs_shape, 1) == get<0>(rhs_shape, 1), "Vectors must have the same size");

        using ResultType = std::common_type_t<typename T::value_t, typename U::value_t>;

        constexpr size_t n0 = get<0>(lhs_shape, 1);
        ResultType sum = 0;
        for (size_t i = 0;i < n0;i++) {
            size_t lhs_idx = T::util::get_effective_idx(i, 0, 0);
            size_t rhs_idx = U::util::get_effective_idx(i, 0, 0);
            sum += lhs(lhs_idx) * rhs(rhs_idx);
        }
        return sum;
    }
}

/**
 * @brief Matrix multiplication of two effective matrices
 * 
 * If lhs has effective shape (n0, n1) and rhs has effective shape (n2, n3), the output 
 * will have effective shape (n0, n3)
 * 
 * n1 must be equal to n2
 * 
 * @tparam T Type of the first matrix (ndarray or subview)
 * @tparam U Type of the second matrix (ndarray or subview)
 * @param lhs First matrix
 * @param rhs Second matrix
 * @return auto Matrix multiplication of the two matrices
*/
TEMPLATE_DECL
KOKKOS_INLINE_FUNCTION decltype(auto) matmul(const T& lhs, const U& rhs) {
    if constexpr (is_ndarray<T>() && is_ndarray<U>()) {
        return matmul(lhs.as_subview_const(), rhs.as_subview_const());
    }
    else if constexpr (is_ndarray<T>() && is_subview<U>()) {
        return matmul(lhs.as_subview_const(), rhs.const_ref());
    }
    else if constexpr (is_subview<T>() && is_ndarray<U>()) {
        return matmul(lhs.const_ref(), rhs.as_subview_const());
    }
    else {
        constexpr auto lhs_shape = T::effective_shape();
        constexpr auto rhs_shape = U::effective_shape();
        constexpr auto lhs_rank = T::effective_rank();
        constexpr auto rhs_rank = U::effective_rank();
        static_assert(lhs_rank == 2 || rhs_rank == 2, "matmul: at least one of the operands must be a matrix");

        using ResultType = std::common_type_t<typename T::value_t, typename U::value_t>;

        if constexpr (lhs_rank < rhs_rank) {
            static_assert(get<0>(lhs_shape, 1) == get<0>(rhs_shape, 1), "matmul: matrices must have compatible shapes");
            constexpr size_t n0 = get<0>(lhs_shape, 1);
            constexpr size_t n1 = get<1>(rhs_shape, 1);
            NdArray<ResultType, n1> out;
            for (size_t j = 0;j < n1;j++) {
                ResultType sum = 0;
                for (size_t i = 0;i < n0;i++) {
                    size_t eff_idx_lhs = T::util::get_effective_idx(i, 0, 0);
                    size_t eff_idx_rhs = U::util::get_effective_idx(i, j, 0);
                    sum += lhs.ref[eff_idx_lhs] * rhs.ref[eff_idx_rhs];
                }
                out(j) = sum;
            }
            return out;
        }
        else if constexpr (lhs_rank == rhs_rank) {
            static_assert(get<1>(lhs_shape, 1) == get<0>(rhs_shape, 1), "matmul: matrices must have compatible shapes");
            constexpr size_t n0 = get<0>(lhs_shape, 1);
            constexpr size_t n1 = get<0>(lhs_shape, 1);
            constexpr size_t n2 = get<1>(rhs_shape, 1);
            NdArray<typename T::value_t, n0, n2> out;
            for (size_t i = 0;i < n0;i++) {
                for (size_t j = 0;j < n2;j++) {
                    ResultType sum = 0;
                    for (size_t k = 0;k < n1;k++) {
                        size_t eff_idx_lhs = T::util::get_effective_idx(i, k, 0);
                        size_t eff_idx_rhs = U::util::get_effective_idx(k, j, 0);
                        sum += lhs.ref[eff_idx_lhs] * rhs.ref[eff_idx_rhs];
                    }
                    out(i, j) = sum;
                }
            }
            return out;
        }
        else {
            static_assert(get<1>(lhs_shape, 1) == get<0>(rhs_shape, 1), "matmul: matrices must have compatible shapes");
            constexpr size_t n0 = get<0>(lhs_shape, 1);
            constexpr size_t n1 = get<1>(lhs_shape, 1);
            constexpr size_t n2 = get<2>(rhs_shape, 1);
            NdArray<typename T::value_t, n0> out;
            for (size_t i = 0;i < n0;i++) {
                ResultType sum = 0;
                for (size_t j = 0;j < n1;j++) {
                    for (size_t k = 0;k < n2;k++) {
                        size_t eff_idx_lhs = T::util::get_effective_idx(i, j, k);
                        size_t eff_idx_rhs = U::util::get_effective_idx(j, k, 0);
                        sum += lhs.ref[eff_idx_lhs] * rhs.ref[eff_idx_rhs];
                    }
                }
                out(i) = sum;
            }
            return out;
        }
    }
}

/**
 * @brief Contraction operation
 * 
 * @tparam T Type of the first vector (ndarray or subview)
 * @tparam U Type of the second matrix (ndarray or subview)
 * @tparam V Type of the third matrix (ndarray or subview)
*/
template<typename T, typename U, typename V,
    typename = typename std::enable_if<is_ndarray<T>() || is_subview<T>()>::type,
    typename = typename std::enable_if<is_ndarray<U>() || is_subview<U>()>::type,
    typename = typename std::enable_if<is_ndarray<V>() || is_subview<V>()>::type
>
KOKKOS_INLINE_FUNCTION decltype(auto) contraction(const T & A, const U & B, const V & C) {
    return dot(A, matmul(B, C));
}
TEMPLATE_DECL
KOKKOS_INLINE_FUNCTION decltype(auto) contraction(const T & A, const U & B) {
    return dot(A, B);
}

/**
 * @brief Wedge product of two effective 3-vectors
 * 
 * @tparam T Type of the first vector (ndarray or subview)
 * @tparam U Type of the second vector (ndarray or subview)
 * @param lhs First vector
 * @param rhs Second vector
 * @param factor Factor to multiply the result by (if factor == 1, then is same as cross-product)
*/
TEMPLATE_DECL
KOKKOS_INLINE_FUNCTION decltype(auto) wedge_product(const T& lhs, const U& rhs, double factor = 1.0) {
    if constexpr (is_ndarray<T>() && is_ndarray<U>()) {
        return wedge_product(lhs.as_subview_const(), rhs.as_subview_const(), factor);
    }
    else if constexpr (is_ndarray<T>() && is_subview<U>()) {
        return wedge_product(lhs.as_subview_const(), rhs.const_ref(), factor);
    }
    else if constexpr (is_subview<T>() && is_ndarray<U>()) {
        return wedge_product(lhs.const_ref(), rhs.as_subview_const(), factor);
    }
    else {
        static_assert(T::effective_rank() == 1 && U::effective_rank() == 1, "Must have two vectors to perform wedge product");
        constexpr auto lhs_shape = T::effective_shape();
        constexpr auto rhs_shape = U::effective_shape();
        static_assert(get<0>(lhs_shape, 1) == 3, "Vectors must have size 3");

        using ResultType = std::common_type_t<typename T::value_t, typename U::value_t>;

        NdArray<ResultType, 3> out;
        size_t lhs_idx0 = T::util::get_effective_idx(0, 0, 0);
        size_t lhs_idx1 = T::util::get_effective_idx(1, 0, 0);
        size_t lhs_idx2 = T::util::get_effective_idx(2, 0, 0);
        size_t rhs_idx0 = U::util::get_effective_idx(0, 0, 0);
        size_t rhs_idx1 = U::util::get_effective_idx(1, 0, 0);
        size_t rhs_idx2 = U::util::get_effective_idx(2, 0, 0);
        factor = 1.0 / factor;
        out(0) = factor * (lhs.ref[lhs_idx1] * rhs.ref[rhs_idx2] - lhs.ref[lhs_idx2] * rhs.ref[rhs_idx1]);
        out(1) = factor * (lhs.ref[lhs_idx2] * rhs.ref[rhs_idx0] - lhs.ref[lhs_idx0] * rhs.ref[rhs_idx2]);
        out(2) = factor * (lhs.ref[lhs_idx0] * rhs.ref[rhs_idx1] - lhs.ref[lhs_idx1] * rhs.ref[rhs_idx0]);
        return out;
    }
}

#undef TEMPLATE_DECL

/**
 * @brief Transpose of an effective matrix or tensor
 * 
 * @tparam T Type of the matrix or tensor (ndarray or subview)
 * @param lhs Matrix or tensor to transpose
 * @return auto Transposed matrix or tensor (copy)
*/
template<typename T, typename = typename std::enable_if<is_ndarray<T>() || is_subview<T>()>::type>
KOKKOS_INLINE_FUNCTION decltype(auto) transpose(const T& lhs) {
    if constexpr (is_ndarray<T>()) {
        return transpose(lhs.as_subview_const());
    }
    else {
        constexpr auto shape = T::effective_shape();
        constexpr auto rank = T::effective_rank();

        constexpr size_t n0 = get<0>(shape, 1);
        constexpr size_t n1 = get<1>(shape, 1);
        constexpr size_t n2 = get<2>(shape, 1);

        if constexpr (rank == 1) {
            return lhs.as_ndarray_cpy();
        }
        else if constexpr (rank == 2) {
            NdArray<typename T::value_t, n1, n0> out;
            for (size_t i = 0;i < n0;i++) {
                for (size_t j = 0;j < n1;j++) {
                    size_t idx = T::util::get_effective_idx(i, j, 0);
                    out(j, i) = lhs.ref[idx];
                }
            }
            return out;
        }
        else {
            NdArray<typename T::value_t, n2, n1, n0> out;
            for (size_t i = 0;i < n0;i++) {
                for (size_t j = 0;j < n1;j++) {
                    for (size_t k = 0;k < n2;k++) {
                        size_t idx = T::util::get_effective_idx(i, j, k);
                        out(k, j, i) = lhs.ref[idx];
                    }
                }
            }
            return out;
        }
    }
}

/**
 * @brief Determinant of a 3x3 effective matrix
 * 
 * @tparam T Type of the matrix (ndarray or subview)
 * @param lhs Matrix to compute
 * @return auto Determinant of the matrix
*/
template<typename T, typename = typename std::enable_if<is_ndarray<T>() || is_subview<T>()>::type>
KOKKOS_INLINE_FUNCTION decltype(auto) det_3x3(const T& lhs) {
    if constexpr (is_ndarray<T>()) {
        return det_3x3(lhs.as_subview_const());
    }
    else {
        constexpr auto shape = T::effective_shape();
        constexpr auto rank = T::effective_rank();
        static_assert(rank == 2, "det_3x3: must have a 3x3 matrix");
        static_assert(get<0>(shape, 1) == 3 && get<1>(shape, 1) == 3, "det_3x3: must have a 3x3 matrix");

        typename T::value_t det = 0;
#define GET(i, j) lhs.ref[T::util::get_effective_idx(i, j, 0)]
        det += GET(0, 0) * (GET(1, 1) * GET(2, 2) - GET(1, 2) * GET(2, 1));
        det -= GET(0, 1) * (GET(1, 0) * GET(2, 2) - GET(1, 2) * GET(2, 0));
        det += GET(0, 2) * (GET(1, 0) * GET(2, 1) - GET(1, 1) * GET(2, 0));
#undef GET
        return det;
    }
}

/**
 * @brief Inverse of a 3x3 effective matrix
 * 
 * @tparam T Type of the matrix (ndarray or subview)
 * @param lhs Matrix to compute
 * @return auto Inverse of the matrix
*/
template<typename T, typename = typename std::enable_if<is_ndarray<T>() || is_subview<T>()>::type>
KOKKOS_INLINE_FUNCTION decltype(auto) mat_3x3_inverse(const T& lhs) {
    if constexpr (is_ndarray<T>()) {
        return mat_3x3_inverse(lhs.as_subview_const());
    }
    else {
        constexpr auto shape = T::effective_shape();
        constexpr auto rank = T::effective_rank();
        static_assert(rank == 2, "det_3x3: must have a 3x3 matrix");
        static_assert(get<0>(shape, 1) == 3 && get<1>(shape, 1) == 3, "det_3x3: must have a 3x3 matrix");

        NdArray<typename T::value_t, 3, 3> out;
        auto det = det_3x3(lhs);
        auto inv_det = 1. / det;
#define GET(i, j) lhs.ref[T::util::get_effective_idx(i, j, 0)]
        out(0, 0) = (GET(1, 1) * GET(2, 2) - GET(1, 2) * GET(2, 1)) * inv_det;
        out(0, 1) = (GET(0, 2) * GET(2, 1) - GET(0, 1) * GET(2, 2)) * inv_det;
        out(0, 2) = (GET(0, 1) * GET(1, 2) - GET(0, 2) * GET(1, 1)) * inv_det;
        out(1, 0) = (GET(1, 2) * GET(2, 0) - GET(1, 0) * GET(2, 2)) * inv_det;
        out(1, 1) = (GET(0, 0) * GET(2, 2) - GET(0, 2) * GET(2, 0)) * inv_det;
        out(1, 2) = (GET(1, 0) * GET(0, 2) - GET(0, 0) * GET(1, 2)) * inv_det;
        out(2, 0) = (GET(1, 0) * GET(2, 1) - GET(2, 0) * GET(1, 1)) * inv_det;
        out(2, 1) = (GET(2, 0) * GET(0, 1) - GET(0, 0) * GET(2, 1)) * inv_det;
        out(2, 2) = (GET(0, 0) * GET(1, 1) - GET(1, 0) * GET(0, 1)) * inv_det;
#undef GET
        return out;
    }
}