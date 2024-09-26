#pragma once
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <string>
#include <mpi.h>
#include "types.h"
#include "util/template_util.h"

inline std::string print_n_chars(int num_spaces, char c = ' ') {
    std::string out;
    for (int i = 0;i < num_spaces;i++)
        out += c;
    return out;
}

template<typename T, typename... Args>
std::string format_line(const T& array, const std::vector<size_t>& sizes, int max_elements, int pre_align, int align_length, const std::string& fmt_args = "", Args... indices) {
    std::string result = print_n_chars(pre_align) + "[";
    size_t length = sizes.at(sizes.size() - 1); 
    if (max_elements < 0) 
        max_elements = length;

    for (size_t i = 0; i < length; ++i) {
        result += fmt::format("{:>" + std::to_string(align_length) + fmt_args + "}", array(indices..., i));
        if (i < length - 1) {
            result += ", ";
        }
        if ((size_t)max_elements < length && i == (size_t)max_elements / 2 - 1) {
            result += "..., ";
            i = length - max_elements / 2 - 1;
        }
    }
    result += "]";
    return result;
}

template<typename T, typename... Args>
std::string format_matrix(const T& array, const std::vector<size_t>& sizes, const std::vector<int>& max_elements, int pre_align, int align_length, const std::string& fmt_args = "", Args... indices) {
    std::string result = print_n_chars(pre_align) + "[";
    size_t dim0 = sizes.at(sizes.size() - 2);
    size_t dim1 = sizes.at(sizes.size() - 1);
    int max_dim0 = max_elements.at(max_elements.size() - 2);
    int max_dim1 = max_elements.at(max_elements.size() - 1);
    if (max_dim0 < 0)
        max_dim0 = dim0;
    if (max_dim1 < 0)
        max_dim1 = dim1;

    for (int j = 0;j < dim0;j++) {
        int pre = 0;
        if (j > 0) {
            pre = pre_align + 1;
        }
        auto line = format_line(array, sizes, max_dim1, pre, align_length, fmt_args, indices..., j);
        result += line;
        if (j < dim0 - 1) {
            result += ",\n";
        }
        if (max_dim0 < dim0 && j == max_dim0 / 2 - 1) {
            result += print_n_chars(pre_align + 1) + "...,\n";
            j = dim0 - max_dim0 / 2 - 1;
        }
    }
    result += "]";
    return result;
}

template<typename T, typename... Args>
std::string format_tensor(const T& array, const std::vector<size_t>& sizes, const std::vector<int>& max_elements, int pre_align, int align_length, const std::string& fmt_args = "", Args... indices) {
    std::string result = "[";
    size_t dim0 = sizes.at(sizes.size() - 3);
    size_t dim1 = sizes.at(sizes.size() - 2);
    size_t dim2 = sizes.at(sizes.size() - 1);
    int max_dim0 = max_elements.at(max_elements.size() - 3);
    int max_dim1 = max_elements.at(max_elements.size() - 2);
    int max_dim2 = max_elements.at(max_elements.size() - 1);
    if (max_dim0 < 0)
        max_dim0 = dim0;
    if (max_dim1 < 0)
        max_dim1 = dim1;
    if (max_dim2 < 0)
        max_dim2 = dim2;

    for (int k = 0;k < dim0;k++) {
        int pre = 0;
        if (k > 0) {
            pre = pre_align + 1;
        }
        auto line = format_matrix(array, sizes, max_elements, pre, align_length, fmt_args, indices..., k);
        result += line;
        if (k < dim0 - 1) {
            result += ",\n\n";
        }
        if (max_dim0 < dim0 && k == max_dim0 / 2 - 1) {
            result += print_n_chars(pre_align + 1) + "...,\n\n";
            k = dim0 - max_dim0 / 2 - 1;
        }
    }
    result += "]";
    return result;
}


#define FORMAT_ARRAY_IMPL(keyword, max_width) \
if keyword (array.rank() == 1) {                                                                                                                                     \
    if (max_width == 0)                                                                                                                                              \
    for(size_t i = 0;i < array.size();i++)                                                                                                                           \
        max_width = std::max(fmt::format("{:" + fmt_args + "}", array(i)).length(), max_width);                                                                      \
                                                                                                                                                                     \
    result += format_line(array, {array.extent(0)}, max_elements, 0, max_width, fmt_args);                                                                           \
}                                                                                                                                                                    \
else if keyword (array.rank() == 2) {                                                                                                                                \
    if (max_width == 0)                                                                                                                                              \
    for(size_t i = 0;i < array.extent(0);i++)                                                                                                                        \
        for(size_t j = 0;j < array.extent(1);j++)                                                                                                                    \
            max_width = std::max(fmt::format("{:" + fmt_args + "}", array(i, j)).length(), max_width);                                                               \
                                                                                                                                                                     \
    result += format_matrix(array, {array.extent(0), array.extent(1)}, {max_elements, max_elements}, 0, max_width, fmt_args);                                        \
}                                                                                                                                                                    \
else if keyword (array.rank() == 3) {                                                                                                                                \
    if (max_width == 0)                                                                                                                                              \
    for(size_t i = 0;i < array.extent(0);i++)                                                                                                                        \
        for(size_t j = 0;j < array.extent(1);j++)                                                                                                                    \
            for (size_t k = 0;k < array.extent(2);k++)                                                                                                               \
                max_width = std::max(fmt::format("{:" + fmt_args + "}", array(i, j, k)).length(), max_width);                                                        \
                                                                                                                                                                     \
    result += format_tensor(array, {array.extent(0), array.extent(1), array.extent(2)}, {max_elements, max_elements, max_elements}, 0, max_width, fmt_args);         \
}                                                                                                                                                                    \
else if keyword (array.rank() == 4) {                                                                                                                                \
    if (max_width == 0)                                                                                                                                              \
    for(size_t i = 0;i < array.extent(0);i++)                                                                                                                        \
        for(size_t j = 0;j < array.extent(1);j++)                                                                                                                    \
            for (size_t k = 0;k < array.extent(2);k++)                                                                                                               \
                for (size_t l = 0;l < array.extent(3);l++)                                                                                                           \
                    max_width = std::max(fmt::format("{:" + fmt_args + "}", array(i, j, k, l)).length(), max_width);                                                 \
                                                                                                                                                                     \
    result += "[";                                                                                                                                                   \
    for (size_t i = 0;i < array.extent(0);i++) {                                                                                                                     \
        result += "Element " + std::to_string(i) + ":\n";                                                                                                            \
        result += format_tensor(array, {array.extent(1), array.extent(2), array.extent(3)}, {max_elements, max_elements, max_elements}, 0, max_width, fmt_args, i);  \
        if (i < array.extent(0) - 1) {                                                                                                                               \
            result += ",\n\n";                                                                                                                                       \
        }                                                                                                                                                            \
        if (max_elements < array.extent(0) && i == max_elements / 2 - 1) {                                                                                           \
            result += "...,\n\n";                                                                                                                                    \
            i = array.extent(0) - max_elements / 2 - 1;                                                                                                              \
        }                                                                                                                                                            \
    }                                                                                                                                                                \
    result += "]";                                                                                                                                                   \
}                                                                                                                                                                    \
else {                                                                                                                                                               \
    throw std::runtime_error("format_array: array rank not supported");                                                                                              \
}

template <typename T, typename = typename std::enable_if<is_kokkos_view<T>()>::type>
std::string format_array(const T& array, const std::string& fmt_args = "", int max_elements = 10, size_t max_el_width = 0) {
    if constexpr (std::is_same_v<typename T::memory_space, Kokkos::HostSpace>) {
        std::string result = array.label() + " :\n";
        if constexpr (has_static_rank<T>()) {
            FORMAT_ARRAY_IMPL(constexpr, max_el_width)
        }
        else {
            FORMAT_ARRAY_IMPL(, max_el_width)
        }
        return result;
    }
    else {
        auto arr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), array);
        return format_array(arr, fmt_args, max_elements, max_el_width);
    }
}
