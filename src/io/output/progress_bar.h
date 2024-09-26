#pragma once
#include "print.h"
#include <string>
std::string progress_bar(double percentage, int width=20) {
    int pos = width * percentage;
    std::string bar = "[";
    for (int i = 0; i < width; ++i) {
        if (i < pos) bar += "=";
        else if (i == pos) bar += ">";
        else bar += " ";
    }
    bar += "] " + std::to_string(int(percentage * 100.0)) + "%";
    return bar;
}