#pragma once

#include "types.h"
#include <fmt/core.h>

// =====================
// Base binary operators
// =====================

#define BINARY_OP(OP, NAME)                                                       \
struct NAME## BinaryOp {                                                          \
    template <typename T, typename U, typename V>                                 \
    KOKKOS_INLINE_FUNCTION void operator()(T& ref, const U& a, const V& b) {      \
        ref = a OP b;                                                             \
    }                                                                             \
};

#define BINARY_FCT_IMPL(OP, NAME)                                                                         \
template<class T, typename U,                                                                             \
    typename = typename std::enable_if<is_ndarray<T>() || is_subview<T>()>::type                          \
>                                                                                                         \
KOKKOS_INLINE_FUNCTION decltype(auto) operator OP(T lhs, const U& rhs) {                                  \
    static_assert(std::is_arithmetic_v<U> || is_ndarray<U>() || is_subview<U>(), "Invalid type for rhs"); \
    if constexpr (std::is_arithmetic_v<U>) {                                                              \
        NAME## BinaryOp op;                                                                               \
        if constexpr (is_ndarray<T>()) {                                                                  \
            return lhs.as_subview_const().template apply_binaryOp<true>(op, rhs);                         \
        }                                                                                                 \
        else {                                                                                            \
            return lhs.const_ref().template apply_binaryOp<true>(op, rhs);                                \
        }                                                                                                 \
    }                                                                                                     \
    else if constexpr (is_ndarray<U>()) {                                                                 \
        NAME## BinaryOp op;                                                                               \
        if constexpr (is_ndarray<T>()) {                                                                  \
            return lhs.as_subview_const().apply_binaryOp(op, rhs.as_subview_const());                     \
        }                                                                                                 \
        else {                                                                                            \
            return lhs.const_ref().apply_binaryOp(op, rhs.as_subview_const());                            \
        }                                                                                                 \
    }                                                                                                     \
    else if constexpr (is_subview<U>()) {                                                                 \
        NAME## BinaryOp op;                                                                               \
        if constexpr (is_ndarray<T>()) {                                                                  \
            return lhs.as_subview_const().apply_binaryOp(op, rhs.const_ref());                            \
        }                                                                                                 \
        else {                                                                                            \
            return lhs.const_ref().apply_binaryOp(op, rhs.const_ref());                                   \
        }                                                                                                 \
    }                                                                                                     \
}                                                                                                         \
template<typename Scalar, class U,                                                                        \
    typename = typename std::enable_if<std::is_scalar<Scalar>::value>::type,                              \
    typename = typename std::enable_if<is_ndarray<U>() || is_subview<U>()>::type                          \
>                                                                                                         \
KOKKOS_INLINE_FUNCTION decltype(auto) operator OP(const Scalar& lhs, const U& rhs) {                      \
    if constexpr (is_ndarray<U>()) {                                                                      \
        NAME## BinaryOp op;                                                                               \
        return rhs.as_subview_const().template apply_binaryOp<false>(op, lhs);                            \
    }                                                                                                     \
    else if constexpr (is_subview<U>()) {                                                                 \
        NAME## BinaryOp op;                                                                               \
        return rhs.const_ref().apply_binaryOp(op, rhs);                                                   \
    }                                                                                                     \
}          


#define DEFINE_FCT(OP, NAME) \
/** @private */              \
BINARY_OP(OP, NAME)          \
/** @private */              \
BINARY_FCT_IMPL(OP, NAME)

DEFINE_FCT(+, Add)
DEFINE_FCT(-, Sub)
DEFINE_FCT(*, Mul)
DEFINE_FCT(/ , Div)
DEFINE_FCT(%, Mod)

DEFINE_FCT(== , Eq)
DEFINE_FCT(!= , Neq)
DEFINE_FCT(> , Gt)
DEFINE_FCT(< , Lt)
DEFINE_FCT(>= , Ge)
DEFINE_FCT(<= , Le)

#undef BINARY_OP
#undef BINARY_FCT_IMPL
#undef DEFINE_FCT

// ======================
// Other binary operators
// ======================

#define BINARY_OP(NAME) \
struct NAME## BinaryOp {                                                     \
    template <typename T, typename U, typename V>                            \
    KOKKOS_INLINE_FUNCTION void operator()(T& ref, const U& a, const V& b) { \
        ref = Kokkos:: NAME (a, b);                                          \
    }                                                                        \
};

#define BINARY_FCT_IMPL(NAME) \
template<typename T, typename U,                                                                                 \
    typename = typename std::enable_if<is_ndarray<T>() || is_subview<T>()>::type,                                \
    typename = typename std::enable_if<is_ndarray<U>() || is_subview<U>() || std::is_arithmetic<U>::value>::type \
>                                                                                                                \
KOKKOS_INLINE_FUNCTION decltype(auto) NAME (const T& a, const U& b) {                                            \
    if constexpr (is_ndarray<T>()) {                                                                             \
        return a.as_subview_const().apply_binaryOp(NAME## BinaryOp{}, b);                                        \
    }                                                                                                            \
    else {                                                                                                       \
        return a.const_ref().apply_binaryOp(NAME## BinaryOp(), b);                                               \
    }                                                                                                            \
}

#define DEFINE_FCT(NAME)  \
/** @private */           \
BINARY_OP(NAME)           \
/** @private */           \
BINARY_FCT_IMPL(NAME)

DEFINE_FCT(max)
DEFINE_FCT(min)
DEFINE_FCT(pow)
DEFINE_FCT(fmod)

#undef BINARY_OP
#undef BINARY_FCT_IMPL