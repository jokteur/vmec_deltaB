/**
 * @file ndarray.h
 * 
 * @brief The goal of this file is to provide a way to do linear algebra (vector, matrix and tensor 
 * operations) with ease of use and performance in mind.
*/
#pragma once

#include <array>
#include <type_traits>
#include "types.h"

#include "template_util.h"
#include "unary_operators.h"
#include "binary_operators.h"

template<typename T, size_t N0, size_t N1, size_t N2>
struct NdArray;

struct SelectAll {};
#define I_ALL SelectAll{}
template<size_t N> struct Index {};
#define IDX(idx) Index<idx>{}

/**
 * @brief This class is for internal use only. It is used for template metaprogramming to determine
 * rank of ndarrays and subviews. This is useful for broadcasting rules.
 * 
 * @tparam N0 Size of the first dimension representing the array
 * @tparam N1 Size of the second dimension representing the array
 * @tparam N2 Size of the third dimension representing the array
 * @tparam Selected0 Index of the first dimension to select (if -1, it means all)
 * @tparam Selected1 Index of the second dimension to select (if -1, it means all)
 * @tparam Selected2 Index of the third dimension to select (if -1, it means all)
 * 
 * If N0 != 1, N1 = N2 = 1, then the rank is 1
 * If N0 != 1, N1 != 1, N2 = 1, then the rank is 2
 * If N0 != 1, N1 != 1, N2 != 1, then the rank is 3
*/
template<typename T, size_t N0, size_t N1, size_t N2, int Selected0, int Selected1, int Selected2>
struct SubViewUtilities {
    constexpr static size_t Ntot = N0 * N1 * N2;

    /**
     * @brief Get the original rank of the array
    */
    static KOKKOS_INLINE_FUNCTION constexpr size_t original_rank() {
        if constexpr (N2 == 1 && N1 == 1 || N0 == 1 && N1 == 1 || N2 == 1 && N1 == 1) return 1;
        else if constexpr (N0 == 1 || N1 == 1 || N2 == 1) return 2;
        else return 3;
    }

    /**
     * @brief Get the effective rank of the array (i.e. the rank after selecting some dimensions)
     * 
     * For example, if the original rank is 3 and Selected0 = -1, Selected1 = 1, Selected2 = -1, then
     * the effective rank is 2
    */
    static KOKKOS_INLINE_FUNCTION constexpr size_t effective_rank() {
        if constexpr (original_rank() == 1) {
            if constexpr (Selected0 != -1) return 0;
            else return 1;
        }
        else if constexpr (original_rank() == 2) {
            if constexpr (Selected0 != -1 && Selected1 != -1) return 0;
            else if constexpr (Selected0 != -1 || Selected1 != -1) return 1;
            else return 2;
        }
        else {
            if constexpr (Selected0 != -1 && Selected1 != -1 && Selected2 != -1) return 0;
            else if constexpr (Selected0 != -1 && Selected1 != -1 || Selected0 != -1 && Selected2 != -1 || Selected1 != -1 && Selected2 != -1) return 1;
            else if constexpr (Selected0 != -1 || Selected1 != -1 || Selected2 != -1) return 2;
            else return 3;
        }
    }

    /**
     * @brief Get the effective shape of the array (i.e. the shape after selecting some dimensions)
    */
    static KOKKOS_INLINE_FUNCTION constexpr decltype(auto) effective_shape() {
        if constexpr (original_rank() == 1) {
            if constexpr (Selected0 != -1) return std::make_tuple();
            else return std::make_tuple(N0);
        }
        else if constexpr (original_rank() == 2) {
            if constexpr (Selected0 != -1 && Selected1 != -1) return std::make_tuple();
            else if constexpr (Selected0 != -1) return std::make_tuple(N1);
            else if constexpr (Selected1 != -1) return std::make_tuple(N0);
            else return std::make_tuple(N0, N1);
        }
        else {
            if constexpr (Selected0 != -1 && Selected1 != -1 && Selected2 != -1) return std::make_tuple();
            else if constexpr (Selected0 != -1 && Selected1 != -1) return std::make_tuple(N2);
            else if constexpr (Selected0 != -1 && Selected2 != -1) return std::make_tuple(N1);
            else if constexpr (Selected1 != -1 && Selected2 != -1) return std::make_tuple(N0);
            else if constexpr (Selected0 != -1) return std::make_tuple(N1, N2);
            else if constexpr (Selected1 != -1) return std::make_tuple(N0, N2);
            else if constexpr (Selected2 != -1) return std::make_tuple(N0, N1);
            else return std::make_tuple(N0, N1, N2);
        }
    }

    /**
     * @brief Check if the broadcasting rules are satisfied between the current subview and another
     * 
     * @tparam Types Types of the shape of the other subview
     * @param rhs_shape Shape of the other subview
     * @return true If the broadcasting rules are satisfied
     * @return false If the broadcasting rules are not satisfied
    */
    template<typename... Types>
    static KOKKOS_INLINE_FUNCTION constexpr bool broadcasting_rules(const std::tuple<Types...>& rhs_shape) {
        constexpr auto lhs_shape = effective_shape();
        constexpr auto lhs_rank = effective_rank();
        constexpr auto rhs_rank = sizeof...(Types);
        if constexpr (sizeof...(Types) == 0 || lhs_rank == 0) {
            return true;
        }
        else if constexpr (lhs_rank + 2 == rhs_rank) {
            return get<0>(lhs_shape) == get<2>(rhs_shape);
        }
        else if constexpr (lhs_rank + 1 == rhs_rank) {
            if constexpr (rhs_rank == 3)
                return get<0>(lhs_shape) == get<1>(rhs_shape) && get<1>(lhs_shape) == get<2>(rhs_shape);
            else
                return get<0>(lhs_shape) == get<1>(rhs_shape);
        }
        else if constexpr (lhs_rank == rhs_rank) {
            return lhs_shape == rhs_shape;
        }
        else if constexpr (rhs_rank + 2 == lhs_rank) {
            return get<0>(rhs_shape) == get<2>(lhs_shape);
        }
        else if constexpr (rhs_rank + 1 == lhs_rank) {
            if constexpr (lhs_rank == 3)
                return get<0>(rhs_shape) == get<1>(lhs_shape) && get<1>(rhs_shape) == get<2>(lhs_shape);
            else
                return get<0>(rhs_shape) == get<1>(lhs_shape);
        }
        else { // should never happen
            return false;
        }
    }

    /**
     * @brief Get the data index in the original array data (which is flat)
     * 
     * Converts i, j, k to a 1D index, taking into account the selected dimensions
    */
    static KOKKOS_INLINE_FUNCTION size_t get_effective_idx(const size_t& i, const size_t& j, const size_t& k) {
        if constexpr (original_rank() == 1) {
            if constexpr (Selected0 != -1) return Selected0;
            else return i;
        }
        else if constexpr (original_rank() == 2) {
            if constexpr (Selected0 != -1 && Selected1 != -1) return Selected0 * N1 + Selected1;
            else if constexpr (Selected0 != -1) return Selected0 * N1 + i;
            else if constexpr (Selected1 != -1) return i * N1 + Selected1;
            else return i * N1 + j;
        }
        else {
            if constexpr (Selected0 != -1 && Selected1 != -1 && Selected2 != -1) return Selected0 * N1 * N2 + Selected1 * N2 + Selected2;
            else if constexpr (Selected0 != -1 && Selected1 != -1) return Selected0 * N2 * N1 + Selected1 * N2 + i; //ok
            else if constexpr (Selected0 != -1 && Selected2 != -1) return Selected0 * N2 * N1 + i * N2 + Selected2; //ok
            else if constexpr (Selected1 != -1 && Selected2 != -1) return i * N2 * N1 + Selected1 * N2 + Selected2; //ok
            else if constexpr (Selected0 != -1) return Selected0 * N2 * N1 + i * N2 + j; //ok
            else if constexpr (Selected1 != -1) return i * N2 * N1 + Selected1 * N2 + j; //ok
            else if constexpr (Selected2 != -1) return i * N2 * N1 + j * N2 + Selected2; //ok
            else return i * N1 * N2 + j * N2 + k;
        }
    }

    /**
     * @brief When broadcasting, we need to reorder the indices of the rhs subview (which may be 
     * of lower or higher rank) to match the lhs subview.
     * 
     * For example, if the lhs subview is of rank 2 and the rhs subview is of rank 3, we need to
     * repeat the index i in the rhs subview to match the lhs subview.
    */
    template<typename... Types>
    KOKKOS_INLINE_FUNCTION void reorder_indices(const std::tuple<Types...>& rhs_shape, size_t& i, size_t& j, size_t& k) const {
        constexpr auto lhs_rank = effective_rank();
        constexpr auto rhs_rank = sizeof...(Types);
        if constexpr (lhs_rank + 2 == rhs_rank) {
            i = k;
        }
        else if constexpr (lhs_rank + 1 == rhs_rank) {
            if constexpr (rhs_rank == 3) {
                i = j; j = k;
            }
            else {
                i = j;
            }
        }
        // lhs_rank == rhs_rank is no op
        else if constexpr (rhs_rank + 2 == lhs_rank) {
            i = k;
        }
        else if constexpr (rhs_rank + 1 == lhs_rank) {
            if constexpr (lhs_rank == 3) {
                i = j; j = k;
            }
            else {
                i = j;
            }
        }
    }
};

/**
 * @brief A const view into an NdArray
 * 
 * @tparam T Type of the elements in the NdArray
 * @tparam N0 Size of the first dimension representing the array
 * @tparam N1 Size of the second dimension representing the array
 * @tparam N2 Size of the third dimension representing the array
 * @tparam Selected0 Index of the first dimension to select (if -1, it means all)
 * @tparam Selected1 Index of the second dimension to select (if -1, it means all)
 * @tparam Selected2 Index of the third dimension to select (if -1, it means all)
*/
template<typename T, size_t N0, size_t N1, size_t N2, int Selected0, int Selected1, int Selected2>
struct SubView_const : public SubViewUtilities<T, N0, N1, N2, Selected0, Selected1, Selected2> {
    using util = SubViewUtilities<T, N0, N1, N2, Selected0, Selected1, Selected2>;
    using value_t = T;
    using subview_flag = bool;
    using subview_const_flag = bool;
    const T* ref;

    /**
     * @brief Construct a new SubView_const object with a reference to the original array
    */
    KOKKOS_INLINE_FUNCTION SubView_const(const T* _ref) : ref(_ref) {}

    /**
     * @brief Get a const reference to the current subview (no op)
    */
    KOKKOS_INLINE_FUNCTION decltype(auto) const_ref() const {
        return *this;
    }

    /**
     * @brief Convert the current subview to an NdArray (copies the data)
    */
    KOKKOS_INLINE_FUNCTION decltype(auto) as_ndarray_cpy() const {
        constexpr auto shape = util::effective_shape();
        constexpr size_t n0 = get<0>(shape, 1);
        constexpr size_t n1 = get<1>(shape, 1);
        constexpr size_t n2 = get<2>(shape, 1);
        
        NdArray<T, n0, n1, n2> out;
        for (size_t i = 0;i < n0;i++) {
            for (size_t j = 0;j < n1;j++) {
                for (size_t k = 0;k < n2;k++) {
                    size_t effective_idx = util::get_effective_idx(i, j, k);
                    out(i, j, k) = ref[effective_idx];
                }
            }
        }
        return out;
    }

    /**
     * @brief Apply a unary operation to the current subview and return the result
     * (does not modifiy the subview)
     * 
     * A unary operation is an operation with only one operand, e.g. cos(a)
     * 
     * @tparam Func Unary operation to apply
     * @param func Unary operation to apply
     * 
     * @return NdArray Result of the unary operation
    */
    template<class Func>
    KOKKOS_INLINE_FUNCTION decltype(auto) apply_unaryOp(Func&& func) const {
        constexpr auto lhs_shape = util::effective_shape();

        constexpr size_t n0 = get<0>(lhs_shape, 1);
        constexpr size_t n1 = get<1>(lhs_shape, 1);
        constexpr size_t n2 = get<2>(lhs_shape, 1);
        NdArray<T, n0, n1, n2> out;
        for (size_t i = 0;i < n0;i++) {
            for (size_t j = 0;j < n1;j++) {
                for (size_t k = 0;k < n2;k++) {
                    size_t effective_lhs_idx = util::get_effective_idx(i, j, k);
                    func(out(i, j, k), ref[effective_lhs_idx]);
                }
            }
        }
        return out;
    }

    /**
     * @brief Apply a binary operation to the current subview and another subview and return the result
     * (does not modifiy the subview)
     * 
     * A binary operation is an operation with two operands, e.g. a + b, or max(a, b)
     * 
     * @tparam Func Binary operation to apply
     * @tparam rhs_type Type of the other subview
     * @param func Binary operation to apply
     * @param subview Other subview
     * @return NdArray Result of the binary operation
    */
    template<class Func, typename rhs_type,
        typename = typename std::enable_if<is_subview<rhs_type>()>::type
    >
    KOKKOS_INLINE_FUNCTION decltype(auto) apply_binaryOp(Func&& func, const rhs_type& subview) const {
        using ResultType = std::common_type_t<T, typename rhs_type::value_t>;

        constexpr auto lhs_shape = util::effective_shape();
        constexpr auto rhs_shape = rhs_type::util::effective_shape();
        constexpr size_t lhs_rank = util::effective_rank();
        constexpr size_t rhs_rank = rhs_type::util::effective_rank();

        static_assert(util::broadcasting_rules(rhs_shape), "Broadcasting rules (between lhs and rhs) are not satisfied");

        if constexpr (lhs_rank == 0 && rhs_rank == 0) {
            ResultType out;
            func(out, ref[util::get_effective_idx(0, 0, 0)], rhs_type::util::get_effective_idx(0, 0, 0));
            return out;
        }
        else if constexpr (lhs_rank >= rhs_rank) {
            constexpr size_t n0 = get<0>(lhs_shape, 1);
            constexpr size_t n1 = get<1>(lhs_shape, 1);
            constexpr size_t n2 = get<2>(lhs_shape, 1);
            NdArray<ResultType, n0, n1, n2> out;
            for (size_t i = 0;i < n0;i++) {
                for (size_t j = 0;j < n1;j++) {
                    for (size_t k = 0;k < n2;k++) {
                        size_t effective_lhs_idx = util::get_effective_idx(i, j, k);
                        size_t ii = i; size_t jj = j; size_t kk = k;
                        util::reorder_indices(rhs_shape, ii, jj, kk);
                        size_t effective_rhs_idx = rhs_type::util::get_effective_idx(ii, jj, kk);
                        func(out(i, j, k), ref[effective_lhs_idx], subview.ref[effective_rhs_idx]);
                    }
                }
            }
            return out;
        }
        else {
            constexpr size_t m0 = get<0>(rhs_shape, 1);
            constexpr size_t m1 = get<1>(rhs_shape, 1);
            constexpr size_t m2 = get<2>(rhs_shape, 1);
            NdArray<ResultType, m0, m1, m2> out;
            for (size_t i = 0;i < m0;i++) {
                for (size_t j = 0;j < m1;j++) {
                    for (size_t k = 0;k < m2;k++) {
                        size_t effective_lhs_idx = util::get_effective_idx(i, j, k);
                        size_t ii = i; size_t jj = j; size_t kk = k;
                        util::reorder_indices(rhs_shape, ii, jj, kk);
                        size_t effective_rhs_idx = rhs_type::util::get_effective_idx(ii, jj, kk);
                        func(out(i, j, k), ref[effective_lhs_idx], subview.ref[effective_rhs_idx]);
                    }
                }
            }
            return out;
        }
    }

    /**
     * @brief Apply a binary operation to the current subview and a scalar and return the result
     * 
     * A binary operation is an operation with two operands, e.g. a + 2, or max(a, 2)
     * 
     * @tparam apply_right If true, the scalar is on the right side of the binary operation, otherwise it is on the left side
     * @tparam Func Binary operation to apply
     * @tparam Scalar Type of the scalar
     * @param func Binary operation to apply
     * @param scalar Scalar
     * @return NdArray Result of the binary operation
    */
    template<bool apply_right, class Func, typename Scalar,
        typename = typename std::enable_if<std::is_arithmetic<Scalar>::value>::type
    >
    KOKKOS_INLINE_FUNCTION decltype(auto) apply_binaryOp(Func&& func, const Scalar& scalar) const {
        constexpr auto lhs_shape = util::effective_shape();
        constexpr size_t lhs_rank = util::effective_rank();
        using ResultType = std::common_type_t<T, Scalar>;

        if constexpr (lhs_rank == 0) {
            ResultType out;
            func(out, ref[util::get_effective_idx(0, 0, 0)], scalar);
            return out;
        }
        else {
            constexpr size_t n0 = get<0>(lhs_shape, 1);
            constexpr size_t n1 = get<1>(lhs_shape, 1);
            constexpr size_t n2 = get<2>(lhs_shape, 1);
            NdArray<ResultType, n0, n1, n2> out;
            for (size_t i = 0;i < n0;i++) {
                for (size_t j = 0;j < n1;j++) {
                    for (size_t k = 0;k < n2;k++) {
                        size_t effective_lhs_idx = util::get_effective_idx(i, j, k);
                        if constexpr (apply_right) {
                            func(out(i, j, k), ref[effective_lhs_idx], scalar);
                        }
                        else {
                            func(out(i, j, k), scalar, ref[effective_lhs_idx]);
                        }
                    }
                }
            }
            return out;
        }
    }

    /**
     * @brief Access the element at (idx0, idx1, idx2)
    */
    KOKKOS_INLINE_FUNCTION T operator()(const size_t& idx0, const size_t& idx1 = SIZE_MAX, const size_t& idx2 = SIZE_MAX) const {
        if constexpr (util::effective_rank() == 3)
            return ref[idx0 * N1 * N2 + idx1 * N2 + idx2];
        else if constexpr (util::effective_rank() == 2)
            return ref[idx0 * N1 + idx1];
        else 
            return ref[idx0];
    }
};

/**
 * @brief A non-const view into an NdArray
 * 
 * @tparam T Type of the elements in the NdArray
 * @tparam N0 Size of the first dimension representing the array
 * @tparam N1 Size of the second dimension representing the array
 * @tparam N2 Size of the third dimension representing the array
 * @tparam Selected0 Index of the first dimension to select (if -1, it means all)
 * @tparam Selected1 Index of the second dimension to select (if -1, it means all)
 * @tparam Selected2 Index of the third dimension to select (if -1, it means all)
*/
template<typename T, size_t N0, size_t N1, size_t N2, int Selected0, int Selected1, int Selected2>
struct SubView : public SubViewUtilities<T, N0, N1, N2, Selected0, Selected1, Selected2> {
    using util = SubViewUtilities<T, N0, N1, N2, Selected0, Selected1, Selected2>;
    using value_t = T;
    using subview_flag = bool;
    T* ref;

    /**
     * @brief Construct a new SubView object with a reference to the original array
    */
    KOKKOS_INLINE_FUNCTION SubView(T* _ref) : ref(_ref) {}

    /**
     * @brief Get a const reference to the current subview
    */
    KOKKOS_INLINE_FUNCTION decltype(auto) const_ref() const {
        return SubView_const<T, N0, N1, N2, Selected0, Selected1, Selected2>(ref);
    }

    /**
     * @brief Convert the current subview to an NdArray (copies the data)
    */
    KOKKOS_INLINE_FUNCTION decltype(auto) as_ndarray_cpy() const {
        constexpr auto shape = util::effective_shape();
        constexpr size_t n0 = get<0>(shape, 1);
        constexpr size_t n1 = get<1>(shape, 1);
        constexpr size_t n2 = get<2>(shape, 1);

        NdArray<T, n0, n1, n2> out;
        for (size_t i = 0;i < n0;i++) {
            for (size_t j = 0;j < n1;j++) {
                for (size_t k = 0;k < n2;k++) {
                    size_t effective_idx = util::get_effective_idx(i, j, k);
                    out(i, j, k) = ref[effective_idx];
                }
            }
        }
        return out;
    }

    /**
     * @brief Apply a binary operation to the current subview with another subview
     * 
     * A self binary operation is an operation with two operands, e.g. a += b
     * 
     * @tparam Func Binary operation to apply
     * @tparam U Type of the other subview
     * @param func Binary operation to apply
     * @param rhs Other subview
    */
    template<class Func, typename U,
        typename = typename std::enable_if<std::is_arithmetic<U>::value || is_subview<U>()>::type
    >
    KOKKOS_INLINE_FUNCTION void self_apply_binaryOp(Func&& func, const U& rhs) {
        constexpr auto lhs_shape = util::effective_shape();

        if constexpr (is_subview<U>()) {
            using ResultType = std::common_type_t<T, typename U::value_t>;
            constexpr auto rhs_shape = U::effective_shape();
            static_assert(util::broadcasting_rules(rhs_shape), "Broadcasting rules (between lhs and rhs) are not satisfied");
            static_assert(util::effective_rank() >= U::util::effective_rank(), "Self operators can only be applied with rhs of lower or equal rank");

            if constexpr (util::effective_rank() == 0 && U::effective_rank() == 0) {
                ResultType out;
                func(out, ref[util::get_effective_idx(0, 0, 0)], U::util::get_effective_idx(0, 0, 0));
            }

            // To avoid self-referencial assignement (i.e. modifying rhs value in the loop), we must make
            // sure that this.ref != rhs.ref. Otherwise, we first must make a copy of rhs first
            if constexpr (std::is_same<T, typename U::value_t>::value) {
                if (this->ref == rhs.ref) {
                    auto rhs_cpy = rhs.as_ndarray_cpy();
                    self_apply_binaryOp(func, rhs_cpy.as_subview_const());
                    return;
                }
            }
        }
        constexpr size_t n0 = get<0>(lhs_shape, 1);
        constexpr size_t n1 = get<1>(lhs_shape, 1);
        constexpr size_t n2 = get<2>(lhs_shape, 1);
        for (size_t i = 0;i < n0;i++) {
            for (size_t j = 0;j < n1;j++) {
                for (size_t k = 0;k < n2;k++) {
                    size_t effective_lhs_idx = util::get_effective_idx(i, j, k);
                    T& lhs = ref[effective_lhs_idx];
                    if constexpr (is_subview<U>()) {
                        size_t ii = i; size_t jj = j; size_t kk = k;
                        util::reorder_indices(U::effective_shape(), ii, jj, kk);
                        size_t effective_rhs_idx = U::util::get_effective_idx(ii, jj, kk);
                        func(lhs, lhs, rhs.ref[effective_rhs_idx]);
                    }
                    else {
                        func(lhs, lhs, rhs);
                    }
                }
            }
        }
    }


#define SELF_SUBVIEW_OP(OP, NAME)                                                                                    \
    template<typename U,                                                                                             \
        typename = typename std::enable_if<std::is_arithmetic<U>::value || is_ndarray<U>() || is_subview<U>()>::type \
    >                                                                                                                \
    KOKKOS_INLINE_FUNCTION decltype(auto) operator OP## =(const U& rhs) {                                            \
        if constexpr (std::is_arithmetic_v<U>) {                                                                     \
            self_apply_binaryOp(NAME## BinaryOp{}, rhs);                                                             \
        }                                                                                                            \
        else if constexpr (is_ndarray<U>()) {                                                                        \
            self_apply_binaryOp(NAME## BinaryOp{}, rhs.as_subview_const());                                          \
        }                                                                                                            \
        else if constexpr (is_subview<U>()) {                                                                        \
            self_apply_binaryOp(NAME## BinaryOp{}, rhs);                                                             \
        }                                                                                                            \
        return *this;                                                                                                \
    }

    /**
     * @brief Self + operator
     * 
     * Adds rhs to the current subview. The rules are the following:
     * - If rhs is a scalar, it will add the scalar to all elements of the subview
     * - If rhs is an ndarray or subview and of the same rank, it will add element-wise the two objects
     * - If rhs is an ndarray or subview of lower rank, it will broadcast the rhs to the current 
     *   subview
     * - If rhs is an ndarray or subview of higher rank, a compilation error will occur
     * 
     * @param rhs The right hand side of the operator (must be a scalar, an ndarray or a subview)
     * 
    */
    SELF_SUBVIEW_OP(+, Add)
    /**
     * @brief Self - operator
     * 
     * Subtracts rhs to the current subview. The rules are the following:
     * - If rhs is a scalar, it will subtract the scalar to all elements of the subview
     * - If rhs is an ndarray or subview and of the same rank, it will subtract element-wise the two objects
     * - If rhs is an ndarray or subview of lower rank, it will broadcast the rhs to the current
    *    subview
    * - If rhs is an ndarray or subview of higher rank, a compilation error will occur
    * 
    * @param rhs The right hand side of the operator (must be a scalar, an ndarray or a subview)
    */
    SELF_SUBVIEW_OP(-, Sub)
    /**
     * @brief Self * operator
     * 
     * Multiplies the current subview by rhs. The rules are the following:
     * - If rhs is a scalar, it will multiply all elements of the subview by the scalar
     * - If rhs is an ndarray or subview and of the same rank, it will multiply element-wise the two objects
     * - If rhs is an ndarray or subview of lower rank, it will broadcast the rhs to the current
     *   subview
     * - If rhs is an ndarray or subview of higher rank, a compilation error will occur
     * 
     * @param rhs The right hand side of the operator (must be a scalar, an ndarray or a subview)
    */
    SELF_SUBVIEW_OP(*, Mul)
    /**
     * @brief Self / operator
     * 
     * Divides the current subview by rhs. The rules are the following:
     * - If rhs is a scalar, it will divide all elements of the subview by the scalar
     * - If rhs is an ndarray or subview and of the same rank, it will divide element-wise the two objects
     * - If rhs is an ndarray or subview of lower rank, it will broadcast the rhs to the current
     *   subview
     * 
     * @param rhs The right hand side of the operator (must be a scalar, an ndarray or a subview)
    */
    SELF_SUBVIEW_OP(/ , Div)
    /**
     * @brief Self % operator
     * 
     * Modulo the current subview by rhs. The rules are the following:
     * - If rhs is a scalar, it will modulo all elements of the subview by the scalar
     * - If rhs is an ndarray or subview and of the same rank, it will modulo element-wise the two objects
     * - If rhs is an ndarray or subview of lower rank, it will broadcast the rhs to the current
     *   subview
     * 
     * @param rhs The right hand side of the operator (must be a scalar, an ndarray or a subview)
    */
    SELF_SUBVIEW_OP(%, Mod)

#undef SELF_SUBVIEW_OP

    /**
     * @brief Assignment operator
     * 
     * Assigns the value of rhs to the current subview. The rules are the following:
     * - If rhs is a scalar, it will assign the scalar to all elements of the subview
     * - If rhs is an ndarray or subview and of the same rank, it will assign element-wise the two objects
     * - If rhs is an ndarray or subview of lower rank, it will broadcast the rhs to the current
     *   subview
     * 
     * @tparam U Type of the right hand side of the operator
     * @param rhs The right hand side of the operator (must be a scalar, an ndarray or a subview)
     * @return SubView& Reference to the current subview
    */
    template<typename U,
        typename = typename std::enable_if<std::is_arithmetic<U>::value || is_ndarray<U>() || is_subview<U>()>::type
    >
    KOKKOS_INLINE_FUNCTION decltype(auto) operator=(const U& rhs) {
        if constexpr (std::is_arithmetic_v<U> || is_subview<U>()) {
            self_apply_binaryOp(SetBinaryOp{}, rhs);
        }
        else if constexpr (is_ndarray<U>()) {
            self_apply_binaryOp(SetBinaryOp{}, rhs.as_subview_const());
        }
        else {
            self_apply_binaryOp(SetBinaryOp{}, rhs);
        }
        return *this;
    }

    /**
     * @brief Access the element at (idx0, idx1, idx2) (const access)
    */
    KOKKOS_INLINE_FUNCTION T operator()(const size_t& idx0, const size_t& idx1 = SIZE_MAX, const size_t& idx2 = SIZE_MAX) const {
        if constexpr (util::effective_rank() == 3)
            return ref[idx0 * N1 * N2 + idx1 * N2 + idx2];
        else if constexpr (util::effective_rank() == 2)
            return ref[idx0 * N1 + idx1];
        else 
            return ref[idx0];
    }
};

/**
 * @brief The NdArray class is a multi-dimensional array that can be used to do linear algebra operations
 * 
 * It is optimised for GPU and CPU, and can be used within Kokkos kernels. Nothing is allocated 
 * dynamically, which allows for the compiler to optimise linear algebra operations. The NdArray 
 * class is a simple wrapper around a raw array of data, and provides a way to access the elements 
 * of the array using the () operator.
 * 
 * @tparam T The type of the elements in the array (usually float or double)
 * @tparam N0 The size of the first dimension
 * @tparam N1 The size of the second dimension (leave to 1 if not used)
 * @tparam N2 The size of the third dimension (leave to 1 if not used)
 * 
 * @subsubsection ex1 Basic example
 * @include examples/ndarray/basic.cpp
 * 
 * @subsubsection ex2 Simple operators
 * @include examples/ndarray/simple_operators.cpp
 * 
 * @subsubsection ex3 Linear algebra
 * @include examples/ndarray/linear_algebra.cpp
 * 
 * @subsubsection ex4 Slicing and broadcasting
 * @include examples/ndarray/slice_and_broadcast.cpp
 * 
 * @subsubsection ex5 Math operators
 * @include examples/ndarray/math_operators.cpp
 * 
 * @subsubsection ex6 Comparison operators
 * @include examples/ndarray/comparison_operators.cpp
*/
template<typename T, size_t N0, size_t N1 = 1, size_t N2 = 1>
struct NdArray {
    constexpr static size_t Ntot = N0 * N1 * N2;
    using value_t = T;
    using Is_NdArray = bool;
    T data[Ntot];

    /**
     * @brief Default constructor that initializes the array to 0
    */
    KOKKOS_INLINE_FUNCTION NdArray() {
        for (size_t i = 0;i < Ntot;i++)
            data[i] = 0;
    }
    /**
     * @brief Copies the data from the given array
    */
    KOKKOS_INLINE_FUNCTION NdArray(const T* _data) {
        for (size_t i = 0;i < Ntot;i++)
            data[i] = _data[i];
    }
    /**
     * @brief Copies the data from the given array
     * 
     * Currently, there are no shallow copies, so this
     * function is the same as
     * @code
     * NdArray<T, N0, N1, N2> a;
     * auto b = a;
     * @endcode
    */
    KOKKOS_INLINE_FUNCTION NdArray<T, N0, N1, N2> deep_copy() const {
        NdArray<T, N0, N1, N2> out;
        for (size_t i = 0;i < Ntot;i++)
            out.data[i] = data[i];
        return out;
    }

    /**
     * @brief Returns the rank of the ndarray
     * 
     * 0 is a scalar, 1 is a vector, 2 is a matrix and 3 is a tensor
    */
    static KOKKOS_INLINE_FUNCTION constexpr size_t rank() {
        if constexpr (N0 == 1 && N1 == 1 && N2 == 1) return 0;
        else if constexpr (N2 == 1 && N1 == 1 || N0 == 1 && N1 == 1 || N2 == 1 && N1 == 1) return 1;
        else if constexpr (N0 == 1 || N1 == 1 || N2 == 1) return 2;
        else return 3;
    }

    /**
     * @brief Returns the array as a const subview
     * 
     * All operators are defined on subview, so this is a way to expose the array to operators.
     * const means that the user won't be able to change the content of the array.
    */
    KOKKOS_INLINE_FUNCTION decltype(auto) as_subview_const() const {
        return SubView_const<T, N0, N1, N2, -1, -1, -1>(data);
    }

    /**
     * @brief Returns the array as a subview
     * 
     * All operators are defined on subview, so this is a way to expose the array to operators.
     * This is the non-const version, so the user can change the content of the array.
    */
    KOKKOS_INLINE_FUNCTION decltype(auto) as_subview() {
        return SubView<T, N0, N1, N2, -1, -1, -1>(data);
    }

    /**
     * @brief Returns the complete ndarray as a subview
    */
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const SelectAll&, const SelectAll&, const SelectAll&) {
        return SubView<T, N0, N1, N2, -1, -1, -1>(data);
    }

    /**
     * @brief Returns a subview of the tensor (matrix)
    */
    template<size_t Idx0>
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const Index<Idx0>&, const SelectAll&, const SelectAll&) {
        static_assert(Idx0 < N0, "Index out of bounds");
        return SubView<T, N0, N1, N2, Idx0, -1, -1>(data);
    }

    /**
     * @brief Returns a subview of the tensor (matrix)
    */
    template<size_t Idx1>
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const SelectAll&, const Index<Idx1>&, const SelectAll&) {
        static_assert(Idx1 < N1, "Index out of bounds");
        return SubView<T, N0, N1, N2, -1, Idx1, -1>(data);
    }

    /**
     * @brief Returns a subview of the tensor (matrix)
    */
    template<size_t Idx2>
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const SelectAll&, const SelectAll&, const Index<Idx2>&) {
        static_assert(Idx2 < N2, "Index out of bounds");
        return SubView<T, N0, N1, N2, -1, -1, Idx2>(data);
    }

    /**
     * @brief Returns a subview of the tensor (vector)
    */
    template<size_t Idx0, size_t Idx1>
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const Index<Idx0>&, const Index<Idx1>&, const SelectAll&) {
        static_assert(Idx0 < N0 && Idx1 < N1, "Index out of bounds");
        return SubView<T, N0, N1, N2, Idx0, Idx1, -1>(data);
    }

    /**
     * @brief Returns a subview of the tensor (vector)
    */
    template<size_t Idx0, size_t Idx2>
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const Index<Idx0>&, const SelectAll&, const Index<Idx2>&) {
        static_assert(Idx0 < N0 && Idx2 < N2, "Index out of bounds");
        return SubView<T, N0, N1, N2, Idx0, -1, Idx2>(data);
    }

    /**
     * @brief Returns a subview of the tensor (vector)
    */
    template<size_t Idx1, size_t Idx2>
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const SelectAll&, const Index<Idx1>&, const Index<Idx2>&) {
        static_assert(Idx1 < N1 && Idx2 < N2, "Index out of bounds");
        return SubView<T, N0, N1, N2, -1, Idx1, Idx2>(data);
    }

    /**
     * @brief Access to the tensor (scalar)
    */
    template<size_t Idx0, size_t Idx1, size_t Idx2>
    KOKKOS_INLINE_FUNCTION T& operator()(const Index<Idx0>&, const Index<Idx1>&, const Index<Idx2>&) {
        static_assert(Idx0 < N0 && Idx1 < N1 && Idx2 < N2, "Index out of bounds");
        return this->operator()(Idx0, Idx1, Idx2);
    }

    /**
     * @brief Returns a subview of the whole ndarray
    */
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const SelectAll&, const SelectAll&) {
        return SubView<T, N0, N1, N2, -1, -1, -1>(data);
    }

    /**
     * @brief Returns a subview (vector or matrix)
    */
    template<size_t Idx0>
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const Index<Idx0>&, const SelectAll&) {
        static_assert(Idx0 < N0, "Index out of bounds");
        return SubView<T, N0, N1, N2, Idx0, -1, -1>(data);
    }

    /**
     * @brief Returns a subview (vector or matrix)
    */
    template<size_t Idx1>
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const SelectAll&, const Index<Idx1>&) {
        static_assert(Idx1 < N1, "Index out of bounds");
        return SubView<T, N0, N1, N2, -1, Idx1, -1>(data);
    }

    /**
     * @brief Returns a subview (scalar or vector)
    */
    template<size_t Idx0, size_t Idx1>
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const Index<Idx0>&, const Index<Idx1>&) {
        static_assert(Idx0 < N0 && Idx1 < N1, "Index out of bounds");
        if constexpr (rank() == 2)
            return this->operator()(Idx0, Idx1);
        else
            return SubView<T, N0, N1, N2, Idx0, Idx1, -1>(data);
    }

    /**
     * @brief Returns a subview (matrix, vector or scalar)
    */
    template<size_t Idx0>
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const Index<Idx0>& idx0) {
        static_assert(Idx0 < N0, "Index out of bounds");
        if constexpr (rank() == 1)
            return this->operator()(Idx0);
        else
            return SubView<T, N0, N1, N2, Idx0, -1, -1>(data);
    }
    /**
     * @brief Returns a subview of the whole ndarray
    */
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const SelectAll&) {
        return SubView<T, N0, N1, N2, -1, -1, -1>(data);
    }

    /**
     * @brief Access the tensor element at (idx0, idx1, idx2) (reference)
    */
    KOKKOS_INLINE_FUNCTION T& operator()(const size_t& idx0, const size_t& idx1, const size_t& idx2) {
        if constexpr (rank() == 3)
            return data[idx0 * N1 * N2 + idx1 * N2 + idx2];
        else if constexpr (rank() == 2)
            return data[idx0 * N1 + idx1];
        else if constexpr (rank() == 1)
            return data[idx0];
    }
    /**
     * @brief Access the tensor element at (idx0, idx1, idx2) (const access)
    */
    KOKKOS_INLINE_FUNCTION T operator()(const size_t& idx0, const size_t& idx1, const size_t& idx2) const {
        if constexpr (rank() == 3)
            return data[idx0 * N1 * N2 + idx1 * N2 + idx2];
        else if constexpr (rank() == 2)
            return data[idx0 * N1 + idx1];
        else if constexpr (rank() == 1)
            return data[idx0];
    }

    /**
     * @brief Access to the matrix (scalar) (reference)
    */
    KOKKOS_INLINE_FUNCTION T& operator()(const size_t& idx0, const size_t& idx1) {
        static_assert(rank() < 3, "Must be less than rank 3 (matrix or vector) to access this operator");
        if constexpr (rank() == 2)
            return data[idx0 * N1 + idx1];
        else if constexpr (rank() == 1)
            return data[idx0];
    }

    /**
     * @brief Access to the matrix (scalar) (const access)
    */
    KOKKOS_INLINE_FUNCTION T operator()(const size_t& idx0, const size_t& idx1) const {
        static_assert(rank() < 3, "Must be less than rank 3 (matrix or vector) to access this operator");
        return data[idx0 * N1 + idx1];
    }

    /**
     * @brief Access to the vector (scalar) (reference)
     */
    KOKKOS_INLINE_FUNCTION T& operator()(const size_t& idx0) {
        static_assert(rank() == 1, "Must be rank 1 (vector) to access this operator");
        return data[idx0];
    }
    /**
     * @brief Access to the vector (scalar) (const access)
     */
    KOKKOS_INLINE_FUNCTION T operator()(const size_t& idx0) const {
        static_assert(rank() == 1, "Must be rank 1 (vector) to access this operator");
        return data[idx0];
    }

#define SELF_NDARRAY_OP(OP, NAME)                                                                                    \
    template<typename U,                                                                                             \
        typename = typename std::enable_if<std::is_arithmetic<U>::value || is_ndarray<U>() || is_subview<U>()>::type \
    >                                                                                                                \
    KOKKOS_INLINE_FUNCTION decltype(auto) operator OP## =(const U& rhs) {                                            \
        if constexpr (std::is_arithmetic_v<U>) {                                                                     \
            as_subview().self_apply_binaryOp(NAME## BinaryOp{}, rhs);                                                \
        }                                                                                                            \
        else if constexpr (is_ndarray<U>()) {                                                                        \
            as_subview().self_apply_binaryOp(NAME## BinaryOp{}, rhs.as_subview_const());                             \
        }                                                                                                            \
        else if constexpr (is_subview<U>()) {                                                                        \
            as_subview().self_apply_binaryOp(NAME## BinaryOp{}, rhs);                                                \
        }                                                                                                            \
        return *this;                                                                                                \
    }

    /**
     * @brief Self + operator
     * 
     * Adds rhs to the current ndarray
     * - If rhs is a scalar, it will add the scalar to all elements of the ndarray
     * - If rhs is an ndarray or subview and of the same rank, it will add element-wise the two ndarrays
     * - If rhs is an ndarray or subview of lower rank, it will broadcast the rhs to the current 
     *   ndarray
     * - If rhs is an ndarray or subview of higher rank, a compilation error will occur
     * 
     * @param rhs The right hand side of the operator (must be a scalar, an ndarray or a subview)
     * 
    */
    SELF_NDARRAY_OP(+, Add)
    /**
     * @brief Self - operator
     * 
     * Subtracts rhs to the current ndarray
     * - If rhs is a scalar, it will subtract the scalar to all elements of the ndarray
     * - If rhs is an ndarray or subview and of the same rank, it will subtract element-wise the two ndarrays
     * - If rhs is an ndarray or subview of lower rank, it will broadcast the rhs to the current
     * - If rhs is an ndarray or subview of higher rank, a compilation error will occur
     * 
     * @param rhs The right hand side of the operator (must be a scalar, an ndarray or a subview)
    */
    SELF_NDARRAY_OP(-, Sub)
    /**
     * @brief Self * operator
     * 
     * Multiplies the current ndarray by rhs
     * - If rhs is a scalar, it will multiply all elements of the ndarray by the scalar
     * - If rhs is an ndarray or subview and of the same rank, it will multiply element-wise the two ndarrays
     * - If rhs is an ndarray or subview of lower rank, it will broadcast the rhs to the current
     * - If rhs is an ndarray or subview of higher rank, a compilation error will occur
     * 
     * @param rhs The right hand side of the operator (must be a scalar, an ndarray or a subview)
    */
    SELF_NDARRAY_OP(*, Mul)
    /**
     * @brief Self / operator
     * 
     * Divides the current ndarray by rhs
     * - If rhs is a scalar, it will divide all elements of the ndarray by the scalar
     * - If rhs is an ndarray or subview and of the same rank, it will divide element-wise the two ndarrays
     * - If rhs is an ndarray or subview of lower rank, it will broadcast the rhs to the current
     * - If rhs is an ndarray or subview of higher rank, a compilation error will occur
     * 
     * @param rhs The right hand side of the operator (must be a scalar, an ndarray or a subview)
    */
    SELF_NDARRAY_OP(/ , Div)
    /**
     * @brief Self % operator
     * 
     * Modulo the current ndarray by rhs
     * - If rhs is a scalar, it will modulo all elements of the ndarray by the scalar
     * - If rhs is an ndarray or subview and of the same rank, it will modulo element-wise the two ndarrays
     * - If rhs is an ndarray or subview of lower rank, it will broadcast the rhs to the current
     * - If rhs is an ndarray or subview of higher rank, a compilation error will occur
     * 
     * @param rhs The right hand side of the operator (must be a scalar, an ndarray or a subview)
    */
    SELF_NDARRAY_OP(%, Mod)
#undef SELF_NDARRAY_OP                                                                                                         

    template<typename U,
        typename = typename std::enable_if<std::is_arithmetic<U>::value || is_ndarray<U>() || is_subview<U>()>::type
    >
    KOKKOS_INLINE_FUNCTION decltype(auto) operator=(const U& rhs) {
        if constexpr (std::is_arithmetic_v<U> || is_subview<U>()) {
            as_subview().self_apply_binaryOp(SetBinaryOp{}, rhs);
        }
        else if constexpr (is_ndarray<U>()) {
            // shallow copy operator
            /// @todo this is not a shallow copy, but a deep copy
            data = rhs.data;
        }
        return *this;
    }
};

#include "generators.h"
#include "print.h"
#include "linalg.h"
