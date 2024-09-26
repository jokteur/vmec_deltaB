#pragma once
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <string>
#include <mpi.h>
#include "print_complex.h"
#include "format_array.h"
#include "logger.h"

template <typename... T>
void print(const std::string& format, T&&... args) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto out = fmt::format(format, std::forward<T>(args)...);
    if (rank == 0)
        fmt::print("{}", out);

    Logger::instance().log(out);
}
template <typename... T>
void println(const std::string& format, T&&... args) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto out = fmt::format(format + "\n", std::forward<T>(args)...);
    if (rank == 0)
        fmt::print("{}", out);

    Logger::instance().log(out);
}

inline std::string hline(int length = 70) {
    std::string out;
    for (int i = 0; i < length; i++) {
        out += "=";
    }
    return out + "\n";
}

inline void print_hline(int length = 70) {
    print(hline(length));
}

inline std::string header(const std::string& title, bool double_header = true, bool center = true, int min_length = 70) {
    size_t text_size = title.size();
    int left_size = 0;
    if (center) {
        left_size = (min_length - text_size) / 2;
    }
    int right_size = min_length - text_size - left_size;
    if (left_size < 0 || right_size < 0) {
        left_size = 0;
        right_size = 0;
        min_length = text_size;
    }
    std::string out;
    out = hline(min_length);
    out += fmt::format("{}{}{}", std::string(left_size, ' '), title, std::string(right_size, ' '));
    if (double_header)  {
        out += "\n";
        out += hline(min_length);  
    } 
    out += "\n";
    return out;
}

inline void print_header(const std::string& title, bool double_header = true, int min_length = 70) {
    print(header(title, double_header, min_length));
}
