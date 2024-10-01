#pragma once

#include <string>
#include <fmt/core.h>
#include <fmt/format.h>
#include "local_array.h"
#include "util/template_util.h"
#include "io/output/print_complex.h"

template<typename T, size_t N0, size_t N1, size_t N2>
std::string fmt_arr(const NdArray<T, N0, N1, N2>& arr, const std::string& fmt_opt = "") {
    using ArrType = NdArray<T, N0, N1, N2>;

    size_t max_string_width = 0;
    for (int i = 0;i < N0 * N1 * N2;i++) {
        std::string string = fmt::format("{:" + fmt_opt + "}", arr.data[i]);
        max_string_width = std::max(max_string_width, string.size());
    }

    std::string out;
    if constexpr (ArrType::rank() == 3) {
        out += fmt::format("[");
        for (size_t i = 0;i < N0;i++) {
            out += fmt::format("[");
            for (size_t j = 0;j < N1;j++) {
                out += fmt::format("[");
                for (size_t k = 0;k < N2;k++) {
                    if constexpr (is_complex<T>())
                        out += fmt::format("{:" + fmt_opt + "}", arr(i, j, k));
                    else
                        out += fmt::format("{:>{}" + fmt_opt + "}", arr(i, j, k), max_string_width);
                    if (k != N2 - 1) out += fmt::format(", ");
                }
                out += fmt::format("]");
                if (j != N1 - 1) out += fmt::format(", \n  ");
            }
            if (i != N0 - 1) out += fmt::format("],\n\n ");
            else out += fmt::format("]");
        }
        out += fmt::format("]");
    }
    else if constexpr (ArrType::rank() == 2) {
        out += fmt::format("[");
        for (size_t i = 0;i < N0;i++) {
            out += fmt::format("[");
            for (size_t j = 0;j < N1;j++) {
                if constexpr (is_complex<T>())
                    out += fmt::format("{:" + fmt_opt + "}", arr(i, j));
                else
                    out += fmt::format("{:>{}" + fmt_opt + "}", arr(i, j), max_string_width);
                if (j != N1 - 1) out += fmt::format(", ");
            }
            if (i != N0 - 1) out += fmt::format("],\n ");
            else out += fmt::format("]");
        }
        out += fmt::format("]");
    }
    else if constexpr (ArrType::rank() == 1) {
        out += fmt::format("[");
        for (size_t i = 0;i < N0;i++) {
            if constexpr (is_complex<T>())
                out += fmt::format("{:" + fmt_opt + "}", arr(i));
            else
                out += fmt::format("{:>{}" + fmt_opt + "}", arr(i), max_string_width);
            if (i != N0 - 1) out += fmt::format(", ");
        }
        out += fmt::format("]");
    }
    return out;
}


template<typename T, typename = typename std::enable_if<is_subview<T>()>::type>
std::string fmt_arr(const T& arr, const std::string& fmt_opt = "") {
    auto cpy = arr.as_ndarray_cpy();
    return fmt_arr(cpy);
}

/// @private
template <typename T>
struct fmt::formatter<T, std::enable_if_t<is_subview<T>() || is_ndarray<T>(), char>> : fmt::formatter<typename T::value_t> {
    mutable std::string fmt_opt;

    constexpr auto parse(format_parse_context& ctx) {
        auto it = ctx.begin(), end = ctx.end();
        fmt_opt.clear();
        while(it != end && *it != '}') {
            if (*it == ':')
                fmt_opt.clear();
            fmt_opt += *it;
            ++it;
        }
        return it;
    }

    auto format(const T& arr, format_context& ctx) const {
        return fmt::format_to(ctx.out(), "{}", fmt_arr(arr, fmt_opt));
    }
};