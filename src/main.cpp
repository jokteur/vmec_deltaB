#include <iostream>
#include "types.h"
#include "io/input/read_h5.h"
#include "io/output/to_file.h"
#include "computation/compute.h"
#include "fourier_array.h"

#define PI 3.1415926535

struct File {
    FourierArray array;
    Kokkos::View<double*, DEVICE> u;
    Kokkos::View<double*, DEVICE> v;
};

File load_fourier_array(const std::string& filename) {
    FourierArray fourier_array;

    HighFive::File file(filename, HighFive::File::ReadOnly);

    fourier_array.xm = to_fixed_sized_rank<1>(H5::fetch_array<Kokkos::DynRankView<int, DEVICE>>(file, "xm"));
    fourier_array.xn = to_fixed_sized_rank<1>(H5::fetch_array<Kokkos::DynRankView<int, DEVICE>>(file, "xn"));

    bool has_cos = file.getAttribute("has_cos").read<bool>();
    bool has_sin = file.getAttribute("has_sin").read<bool>();
    if (has_cos) {
        fourier_array.has_cos = true;
        fourier_array.cos_coefficient = to_fixed_sized_rank<2>(H5::fetch_array<Kokkos::DynRankView<double, DEVICE>>(file, "cos_coefficient"));
    }
    if (has_sin) {
        fourier_array.has_sin = true;
        fourier_array.sin_coefficient = to_fixed_sized_rank<2>(H5::fetch_array<Kokkos::DynRankView<double, DEVICE>>(file, "sin_coefficient"));
    }

    File out;
    out.array = fourier_array;
    size_t u_size = file.getAttribute("u_size").read<size_t>();
    size_t v_size = file.getAttribute("v_size").read<size_t>();
    out.u = linspace<Kokkos::View<double*, DEVICE>>(0.0, 2.0 * PI, u_size);
    out.v = linspace<Kokkos::View<double*, DEVICE>>(0.0, 2.0 * PI, v_size);

    return out;
}

FourierArray find_delta_coefficients(FourierArray& array, const Kokkos::View<double*, DEVICE>& u, const Kokkos::View<double*, DEVICE>& v) {
    FourierArray new_array = prepare_for_inverse_fourier(array.xm, array.xn, array.num_surfaces(), array.has_cos, array.has_sin);

    for (int s_idx = 0;s_idx < array.num_surfaces();s_idx++) {
        Kokkos::fence();
        Kokkos::Timer timer;
        auto field_3d = evaluate_surface(array, s_idx, u, v, false);
        auto field_2d = evaluate_surface(array, s_idx, u, v, true);
        auto delta = field_3d - field_2d;

        inverse_fourier_at_surface(new_array, delta, u, v, s_idx);
        Kokkos::fence();
        double time = timer.seconds();
        println("s: {}, Time to find delta: {}", s_idx, time);
    }

    return new_array;
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <filename> <out_file>" << std::endl;
        return 1;
    }
    int mpi_rank, mpi_size;

    // initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    Kokkos::initialize(argc, argv);
    {
        auto file = load_fourier_array(argv[1]);
        auto array = find_delta_coefficients(file.array, file.u, file.v);

        HighFive::File out_file(argv[2], HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);
        H5::write_dataset(out_file, out_file, "xm", array.xm);
        H5::write_dataset(out_file, out_file, "xn", array.xn);
        out_file.createAttribute<bool>("has_cos", HighFive::DataSpace::From(true)).write(array.has_cos);
        out_file.createAttribute<bool>("has_sin", HighFive::DataSpace::From(true)).write(array.has_sin);
        if (array.has_cos) {
            H5::write_dataset(out_file, out_file, "cos_coefficient", array.cos_coefficient);
        }
        if (array.has_sin) {
            H5::write_dataset(out_file, out_file, "sin_coefficient", array.sin_coefficient);
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}