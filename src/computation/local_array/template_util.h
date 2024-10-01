#pragma once
#include "types.h"
#include "util/template_util.h"
#include <type_traits>

/** @private */
namespace detail {
    /// @private
    template <typename T, typename = void>
    struct is_ndarray : std::false_type {
    };

    /// @private
    template <typename T>
    struct is_ndarray<T, std::void_t<typename T::Is_NdArray>> : std::true_type {
    };

    /// @private
    template <typename T, typename = void>
    struct is_subview : std::false_type {
    };

    /// @private
    template <typename T>
    struct is_subview<T, std::void_t<typename T::subview_flag>> : std::true_type {
    };

    /// @private
    template <typename T, typename = void>
    struct is_subview_const : std::false_type {
    };

    /// @private
    template <typename T>
    struct is_subview_const<T, std::void_t<typename T::subview_const_flag>> : std::true_type {
    };
}

/**
 * @brief Determine if a type is an NdArray
 *
 * @tparam T Type to test
 */
template <typename T> constexpr bool is_ndarray() {
    return detail::is_ndarray<typename remove_cvref<T>::type>::value;
}
/**
 * @brief Determine if a type is a SubView
 *
 * @tparam T Type to test
 */
template <typename T> constexpr bool is_subview() {
    return detail::is_subview<typename remove_cvref<T>::type>::value;
}

/**
 * @brief Determine if a type is a const SubView
 *
 * @tparam T Type to test
 */
template <typename T> constexpr bool is_subview_const() {
    return detail::is_subview_const<typename remove_cvref<T>::type>::value;
}
