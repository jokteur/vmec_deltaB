#pragma once

#include <vector>
#include <string>

#include "types.h"

std::string pretty_print(const std::vector<double>& a);
/**
 * @brief Print a 1D array
*/
template<typename T>
std::string print_array(const T& a, int start = 0, int end = -1);

template<typename T>
std::string print_dimensions(const T& a);

#include "pretty_print.impl.h"