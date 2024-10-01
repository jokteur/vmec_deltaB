#pragma once

#include <exception>
#include <limits>
#include "types.h"
#include "util/template_util.h"

template<typename T>
struct ValueWrapper {
    T value;
    ValueWrapper(const T& value) : value(value) { }
    template<typename... Args>
    KOKKOS_INLINE_FUNCTION T operator()(Args...) const { return value; }
};
// Is Kokkos or numeric
namespace detail {
    /// @private
    template<typename T>
    struct is_numeric_or_kokkos_view {
        static constexpr bool value = is_kokkos_view<T>() || std::is_arithmetic<T>::value;
    };
}
template <typename T> constexpr bool is_numeric_or_kokkos_view() {
    return detail::is_numeric_or_kokkos_view<typename remove_cvref<T>::type>::value;
}

template<typename T>
decltype(auto) __get_compatible_view(const T& input) {
    if constexpr (is_kokkos_view<T>()) {
        return input;
    }
    else if constexpr (std::is_arithmetic<T>::value) {
        return ValueWrapper<T>(input);
    }
    else {
        static_assert(is_numeric_or_kokkos_view<T>(), "Type must be numeric or a Kokkos view");
    }
}
template<typename T, typename U>
size_t __get_extent(size_t i, const T& input, const U& output) {
    if constexpr (is_kokkos_view<T>()) {
        return input.extent(i);
    }
    else if constexpr (is_kokkos_view<U>()) {
        return output.extent(i);
    }
    else {
        static_assert(is_numeric_or_kokkos_view<T>(), "One of the type must be a kokkos view");
    }
}
template<typename T, typename U>
decltype(auto) __get_label(const T& input, const U& output) {
    if constexpr (is_kokkos_view<T>()) {
        return input.label();
    }
    else if constexpr (is_kokkos_view<U>()) {
        return output.label();
    }
    else {
        static_assert(is_numeric_or_kokkos_view<T>(), "One of the type must be a kokkos view");
    }
}


#define BASIC_OPERATOR(op, name) \
/** @private */ \
template<typename Lhs, typename Rhs, typename Output>                                                                  \
struct name## Functor {                                                                                                \
    /* The functor is there to avoid the error error: The enclosing parent function ("fct") */                         \
    /* for an extended __host__ __device__ lambda must not have deduced return type*/                                  \
    Lhs lhs;                                                                                                           \
    Rhs rhs;                                                                                                           \
    Output output;                                                                                                     \
    name## Functor(const Lhs& lhs, const Rhs& rhs, Output& output) : lhs(lhs), rhs(rhs), output(output) { }            \
    KOKKOS_INLINE_FUNCTION void operator()(const size_t& i) const {                                                    \
            output(i) = lhs(i) op rhs(i);                                                                              \
    }                                                                                                                  \
    KOKKOS_INLINE_FUNCTION void operator()(const size_t& i, const size_t& j) const {                                   \
            output(i, j) = lhs(i, j) op rhs(i, j);                                                                     \
    }                                                                                                                  \
    KOKKOS_INLINE_FUNCTION void operator()(const size_t& i, const size_t& j, const size_t& k) const {                  \
            output(i, j, k) = lhs(i, j, k) op rhs(i, j, k);                                                            \
    }                                                                                                                  \
    KOKKOS_INLINE_FUNCTION void operator()(const size_t& i, const size_t& j, const size_t& k, const size_t& l) const { \
            output(i, j, k, l) = lhs(i, j, k, l) op rhs(i, j, k, l);                                                   \
    }                                                                                                                  \
};    /** @private */                                                                                                  \
template<int rank, typename T, typename U,                                                                             \
    typename = typename std::enable_if<is_numeric_or_kokkos_view<U>() && is_numeric_or_kokkos_view<U>()>::type>        \
decltype(auto) name  (const T& lhs, const U& rhs) {                                                                    \
    if constexpr (is_kokkos_view<T>() && is_kokkos_view<U>()) {                                                        \
        if constexpr (has_static_rank<T>())                                                                            \
            static_assert(rank == T::Rank, "Rank must be the same as the first argument");                             \
        if constexpr (has_static_rank<U>())                                                                            \
            static_assert(rank == U::Rank, "Rank must be the same as the second argument");                            \
        if(lhs.rank() != rhs.rank())                                                                                   \
            throw std::runtime_error("Both views must have the same rank");                                            \
    }                                                                                                                  \
    if constexpr (std::is_arithmetic<T>::value && std::is_arithmetic<U>::value) {                                      \
        return lhs op rhs;                                                                                             \
    }                                                                                                                  \
    /* static_assert(T::memory_space == U::memory_space, "Both views must have the same memory space"); */             \
    using space = typename T::memory_space;                                                                            \
    using type = typename T::value_type;                                                                               \
    auto lhs_ = __get_compatible_view(lhs);                                                                            \
    auto rhs_ = __get_compatible_view(rhs);                                                                            \
    auto label = __get_label(lhs, rhs);                                                                                \
    if constexpr (rank == 1) {                                                                                         \
        auto extent0 = __get_extent(0, lhs, rhs);                                                                      \
        Kokkos::View<type*, space> output(label, extent0);                                                             \
        auto policy = Kokkos::RangePolicy<typename T::execution_space>(0, extent0);                                    \
        Kokkos::parallel_for("op", policy,                                                                             \
            name## Functor<decltype(lhs_), decltype(rhs_), decltype(output)>(lhs_, rhs_, output));                     \
        return output;                                                                                                 \
    }                                                                                                                  \
    else if constexpr (rank == 2) {                                                                                    \
        auto extent0 = __get_extent(0, lhs, rhs);                                                                      \
        auto extent1 = __get_extent(1, lhs, rhs);                                                                      \
        Kokkos::View<type**, space> output(label, extent0, extent1);                                                   \
        auto policy = Kokkos::MDRangePolicy<typename T::execution_space, Kokkos::Rank<2>>                              \
            ({0,0},{extent0, extent1});                                                                                \
        Kokkos::parallel_for("op", policy,                                                                             \
            name## Functor<decltype(lhs_), decltype(rhs_), decltype(output)>(lhs_, rhs_, output));                     \
        return output;                                                                                                 \
    }                                                                                                                  \
    else if constexpr (rank == 3) {                                                                                    \
        auto extent0 = __get_extent(0, lhs, rhs);                                                                      \
        auto extent1 = __get_extent(1, lhs, rhs);                                                                      \
        auto extent2 = __get_extent(2, lhs, rhs);                                                                      \
        Kokkos::View<type***, space> output(label, extent0, extent1, extent2);                                         \
        auto policy = Kokkos::MDRangePolicy<typename T::execution_space, Kokkos::Rank<3>>                              \
            ({0,0,0},{extent0, extent1, extent2});                                                                     \
        Kokkos::parallel_for("op", policy,                                                                             \
            name## Functor<decltype(lhs_), decltype(rhs_), decltype(output)>(lhs_, rhs_, output));                     \
        return output;                                                                                                 \
    }                                                                                                                  \
    else if constexpr (rank == 4) {                                                                                    \
        auto extent0 = __get_extent(0, lhs, rhs);                                                                      \
        auto extent1 = __get_extent(1, lhs, rhs);                                                                      \
        auto extent2 = __get_extent(2, lhs, rhs);                                                                      \
        auto extent3 = __get_extent(3, lhs, rhs);                                                                      \
        Kokkos::View<type****, space> output(label, extent0, extent1, extent2, extent3);                               \
        auto policy = Kokkos::MDRangePolicy<typename T::execution_space, Kokkos::Rank<4>>                              \
            ({0,0,0,0},{extent0, extent1, extent2, extent3});                                                          \
        Kokkos::parallel_for("op", policy,                                                                             \
            name## Functor<decltype(lhs_), decltype(rhs_), decltype(output)>(lhs_, rhs_, output));                     \
        return output;                                                                                                 \
    }                                                                                                                  \
    else  {                                                                                                            \
        static_assert(rank < 5, "Rank must be between 1 and 4");                                                       \
    }                                                                                                                  \
}  /** @private */                                                                                                     \
template<typename T, typename U, typename = typename std::enable_if_t<                                                 \
    is_kokkos_view<T>() && is_numeric_or_kokkos_view<U>() || is_kokkos_view<U>() && is_numeric_or_kokkos_view<U>()>>   \
decltype(auto) operator op (const T& lhs, const U& rhs) {                                                              \
    if constexpr (has_static_rank<T>()) {                                                                              \
        return name <T::rank()>(lhs, rhs);                                                                             \
    }                                                                                                                  \
    else if constexpr (has_static_rank<U>()) {                                                                         \
        return name <U::rank()>(lhs, rhs);                                                                             \
    }                                                                                                                  \
    else {                                                                                                             \
        static_assert(has_static_rank<T>() || has_static_rank<U>(), "Cannot use operator for two dynrankview");        \
    }                                                                                                                  \
}                                                                                                                      

BASIC_OPERATOR(+, add)
BASIC_OPERATOR(-, subtract)
BASIC_OPERATOR(*, multiply)
BASIC_OPERATOR(/ , divide)


#undef BASIC_OPERATOR

template<int rank, typename Lhs, typename Rhs>
void assign(Lhs& lhs, const Rhs& rhs) {
    if constexpr (is_kokkos_view<Lhs>() && is_kokkos_view<Rhs>()) {
        if(lhs.rank() != rhs.rank())
            throw std::runtime_error("assign: both views must have the same rank");
        if(lhs.extent(0) != rhs.extent(0) || lhs.extent(1) != rhs.extent(1)
            || lhs.extent(2) != rhs.extent(2) || lhs.extent(3) != rhs.extent(3))
            throw std::runtime_error("assign: both views must have the same extent");
    }
    else {
        static_assert(is_kokkos_view<Lhs>(), "assign: type must be numeric or a Kokkos view");
    }
    using space = typename Lhs::memory_space;
    using type = typename Lhs::value_type;
    auto label = __get_label(lhs, rhs);
    auto rhs_ = __get_compatible_view(rhs);
    auto lhs_ = __get_compatible_view(lhs);
    if constexpr (rank == 1) {
        auto extent0 = lhs.extent(0);
        auto policy = Kokkos::RangePolicy<typename Lhs::execution_space>(0, extent0);
        Kokkos::parallel_for("assign", policy, KOKKOS_LAMBDA(const size_t& i) {
            lhs_(i) = rhs_(i);
        });
    }
    else if constexpr (rank == 2) {
        auto extent0 = lhs.extent(0);
        auto extent1 = lhs.extent(1);
        auto policy = Kokkos::MDRangePolicy<typename Lhs::execution_space, Kokkos::Rank<2>>
            ({0,0},{extent0, extent1});
        Kokkos::parallel_for("assign", policy, KOKKOS_LAMBDA(const size_t& i, const size_t& j) {
            lhs_(i, j) = rhs_(i, j);
        });
    }
    else if constexpr (rank == 3) {
        auto extent0 = lhs.extent(0);
        auto extent1 = lhs.extent(1);
        auto extent2 = lhs.extent(2);
        auto policy = Kokkos::MDRangePolicy<typename Lhs::execution_space, Kokkos::Rank<3>>
            ({0,0,0},{extent0, extent1, extent2});
        Kokkos::parallel_for("assign", policy, KOKKOS_LAMBDA(const size_t& i, const size_t& j, const size_t& k) {
            lhs_(i, j, k) = rhs_(i, j, k);
        });
    }
    else if constexpr (rank == 4) {
        auto extent0 = lhs.extent(0);
        auto extent1 = lhs.extent(1);
        auto extent2 = lhs.extent(2);
        auto extent3 = lhs.extent(3);
        auto policy = Kokkos::MDRangePolicy<typename Lhs::execution_space, Kokkos::Rank<4>>
            ({0,0,0,0},{extent0, extent1, extent2, extent3});
        Kokkos::parallel_for("assign", policy, KOKKOS_LAMBDA(const size_t& i, const size_t& j, const size_t& k, const size_t& l) {
            lhs_(i, j, k, l) = rhs_(i, j, k, l);
        });
    }
    else {
        throw std::runtime_error("assign: Rank must be between 1 and 4");
    }
}