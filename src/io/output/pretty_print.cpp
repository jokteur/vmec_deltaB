#include "pretty_print.h"

#include <sstream>
#include <iostream>

std::string print_array(const std::vector<double>& a) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < a.size(); ++i) {
        ss << a[i];
        if (i != a.size() - 1) {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}