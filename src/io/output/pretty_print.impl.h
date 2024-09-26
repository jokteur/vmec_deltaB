#include "pretty_print.h"

template<typename T>
std::string print_array(const T& a, int start, int end) {
    std::stringstream ss;

    if (end == -1)
        end = a.extent(0);

    auto sub_view = Kokkos::subview(a, SLICE(start, end));
    auto host_view = Kokkos::create_mirror_view(sub_view);
    Kokkos::deep_copy(host_view, sub_view);

    size_t ustart = start;
    size_t uend = end;

    ss << "[";
    if (start > 0)
        ss << "..., ";
    for (size_t i = ustart; i < uend; ++i) {
        ss << host_view(i);
        if (i != uend - 1) {
            ss << ", ";
        }
    }
    if (uend < a.extent(0))
        ss << ", ...";
    ss << "]";
    return ss.str();
}

template<typename T>
std::string print_dimensions(const T& a) {
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < a.rank; ++i) {
        ss << a.extent(i);
        if (i != a.rank - 1) {
            ss << ", ";
        }
    }
    ss << ")";
    return ss.str();
}