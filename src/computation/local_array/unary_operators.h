#pragma once

#include "types.h"

/// @private
struct SetUnaryOp {
    template <typename T, typename U>
    KOKKOS_INLINE_FUNCTION void operator()(T& ref, const U& a) {
        ref = a;
    }
};

/// @private
struct SetBinaryOp {
    template <typename T, typename U, typename V>
    KOKKOS_INLINE_FUNCTION void operator()(T& ref, const U& a, const V& b) {
        ref = b;
    }
};

/// @private
struct signUnaryOp {
    template <typename T, typename U>
    KOKKOS_INLINE_FUNCTION void operator()(T& ref, const U& a) {
        ref = (T)(a > 0.) - (T)(a < 0.);
    }
};

#define UNARY_OP(NAME) \
struct NAME## UnaryOp {                                          \
    template <typename T, typename U>                            \
    KOKKOS_INLINE_FUNCTION void operator()(T& ref, const U& a) { \
        ref = Kokkos:: NAME(a);                                  \
    }                                                            \
};

#define UNARY_FCT_IMPL(NAME) \
template<typename T,                                                             \
    typename = typename std::enable_if<is_ndarray<T>() || is_subview<T>()>::type \
>                                                                                \
KOKKOS_INLINE_FUNCTION decltype(auto) NAME (const T& a) {                        \
    if constexpr (is_ndarray<T>()) {                                             \
        return a.as_subview_const().apply_unaryOp(NAME## UnaryOp{});             \
    }                                                                            \
    else {                                                                       \
        return a.const_ref().apply_unaryOp(NAME## UnaryOp());                    \
    }                                                                            \
}

#define DEFINE_FCT(NAME) \
/** @private */          \
UNARY_OP(NAME)           \
/** @private */          \
UNARY_FCT_IMPL(NAME)

DEFINE_FCT(sin)
DEFINE_FCT(cos)
DEFINE_FCT(tan)
DEFINE_FCT(asin)
DEFINE_FCT(acos)
DEFINE_FCT(atan)
DEFINE_FCT(sinh)
DEFINE_FCT(cosh)
DEFINE_FCT(tanh)
DEFINE_FCT(asinh)
DEFINE_FCT(acosh)
DEFINE_FCT(atanh)
DEFINE_FCT(exp)
DEFINE_FCT(log)
DEFINE_FCT(sqrt)
DEFINE_FCT(abs)
DEFINE_FCT(floor)
DEFINE_FCT(ceil)
DEFINE_FCT(erf)
UNARY_FCT_IMPL(sign)

#undef DEFINE_FCT
#undef UNARY_FCT_IMPL
#undef UNARY_OP

