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

    // For evaluation
    Kokkos::View<double*, DEVICE> u;
    Kokkos::View<double*, DEVICE> v;
    size_t s_idx;
    Kokkos::View<double**, DEVICE> result;
    bool kill_3D_modes = false;

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

    KOKKOS_INLINE_FUNCTION void operator()(const int& u_idx, const int& v_idx) const {
        int num_n = xn.extent(0);

        double sum = 0.0;
        double uu = u(u_idx);
        double vv = v(v_idx);
        for (int nm = 0; nm < num_n; nm++) {
            int m = xm(nm);
            int n = xn(nm);
            if (kill_3D_modes && m != 0)
                continue;
            if (has_cos) {
                sum += cos_coefficient(s_idx, nm) * Kokkos::cos(m * uu - n * vv);
            }
            if (has_sin) {
                sum += sin_coefficient(s_idx, nm) * Kokkos::sin(m * uu - n * vv);
            }
        }
        result(u_idx, v_idx) = sum;
    }
};

Kokkos::View<double**, DEVICE> evaluate_surface(FourierArray& array, int s_idx, const Kokkos::View<double*, DEVICE>& u, const Kokkos::View<double*, DEVICE>& v, bool kill_3D_modes = false) {
    Kokkos::View<double**, DEVICE> result("result", u.extent(0), v.extent(0));
    array.result = result;
    array.s_idx = s_idx;
    array.u = u;
    array.v = v;
    array.kill_3D_modes = kill_3D_modes;

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { u.extent(0), v.extent(0) });
    Kokkos::parallel_for("evaluate", policy, array);
    return array.result;
}
