#pragma once
#include "types.h"
#include "io/input/read_h5.h"
#include "computation/compute.h"

#define PI 3.1415926535

struct FourierArray {
    Kokkos::View<int*, DEVICE> xm;
    Kokkos::View<int*, DEVICE> xn;
    Kokkos::View<double**, DEVICE> cos_coefficient;
    Kokkos::View<double**, DEVICE> sin_coefficient;

    bool has_cos = false;
    bool has_sin = false;

    int num_surfaces() const {
        if (has_sin) {
            return sin_coefficient.extent(0);
        }
        else {
            return cos_coefficient.extent(0);
        }
    }
};

Kokkos::View<double**, DEVICE> evaluate_surface(FourierArray& array, int s_idx, const Kokkos::View<double*, DEVICE>& u, const Kokkos::View<double*, DEVICE>& v, bool kill_3D_modes = false) {
    Kokkos::View<double**, DEVICE> result("result", u.extent(0), v.extent(0));

    // Create parallel policy
    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { u.extent(0), v.extent(0) });
    Kokkos::parallel_for("evaluate", policy, KOKKOS_LAMBDA(const int& u_idx, const int& v_idx) {
        int num_n = array.xn.extent(0);

        double sum = 0.0;
        double uu = u(u_idx);
        double vv = v(v_idx);
        for (int nm = 0; nm < num_n; nm++) {
            int m = array.xm(nm);
            int n = array.xn(nm);
            if (kill_3D_modes && m != 0)
                continue;
            if (array.has_cos) {
                sum += array.cos_coefficient(s_idx, nm) * Kokkos::cos(m * uu - n * vv);
            }
            if (array.has_sin) {
                sum += array.sin_coefficient(s_idx, nm) * Kokkos::sin(m * uu - n * vv);
            }
        }
        result(u_idx, v_idx) = sum;
    });
    return result;
}

FourierArray prepare_for_inverse_fourier(const Kokkos::View<int*, DEVICE>& xm, const Kokkos::View<int*, DEVICE>& xn, int num_surface, bool has_cos, bool has_sin) {
    FourierArray result;
    result.xm = xm;
    result.xn = xn;
    result.has_cos = has_cos;
    result.has_sin = has_sin;
    if (has_cos) {
        result.cos_coefficient = Kokkos::View<double**, DEVICE>("cos_coefficient", num_surface, xm.extent(0));
    }
    if (has_sin) {
        result.sin_coefficient = Kokkos::View<double**, DEVICE>("sin_coefficient", num_surface, xm.extent(0));
    }
    return result;
}

void inverse_fourier_at_surface(
    FourierArray& array,
    const Kokkos::View<double**, DEVICE>& field,
    const Kokkos::View<double*, DEVICE>& u,
    const Kokkos::View<double*, DEVICE>& v,
    int s_idx) {

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { u.extent(0), v.extent(0) });
    auto cos_coefficient = array.cos_coefficient;
    auto sin_coefficient = array.sin_coefficient;
    auto has_cos = array.has_cos;
    auto has_sin = array.has_sin;
    auto xm = array.xm;
    auto xn = array.xn;

    for (int nm = 0; nm < array.xn.extent(0); nm++) {
        double cos_mode, sin_mode;
        Kokkos::parallel_reduce("inverse_fourier", policy, KOKKOS_LAMBDA(const int& u_idx, const int& v_idx, double& cos_sum, double& sin_sum) {
            double uu = u(u_idx);
            double vv = v(v_idx);
            int m = xm(nm);
            int n = xn(nm);
            double value = field(u_idx, v_idx);
            if (has_cos) {
                cos_sum += value * cos_coefficient(s_idx, nm) * Kokkos::cos(m * uu - n * vv);
            }
            if (has_sin) {
                sin_sum += value * sin_coefficient(s_idx, nm) * Kokkos::sin(m * uu - n * vv);
            }
        }, cos_mode, sin_mode);

        Kokkos::parallel_for("assign", 1, KOKKOS_LAMBDA(const int&) {
            if (has_cos) {
                cos_coefficient(s_idx, nm) = cos_mode;
                if (xm(nm) == 0)
                    cos_coefficient(s_idx, nm) /= 2.0;
            }
            if (has_sin) {
                sin_coefficient(s_idx, nm) = sin_mode;
                if (xm(nm) == 0)
                    sin_coefficient(s_idx, nm) /= 2.0;
            }
        });
    }
}