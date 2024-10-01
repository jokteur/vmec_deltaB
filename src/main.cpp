#include <iostream>
#include "types.h"
#include "io/input/read_h5.h"
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
        Kokkos::Timer timer;
        for (int s = 0;s < file.array.num_surfaces();s++) {
            Kokkos::fence();
            timer.reset();
            auto result = evaluate_surface(file.array, s, file.u, file.v);
            Kokkos::fence();
            double time = timer.seconds();
            auto result_cpu = to_cpu<Kokkos::View<double**, HOST>>(result);
            // println("s: {}, Array: {}", s, format_array(result_cpu));
            println("s: {}, Time to evaluate: {}", s, time);
        }

        // HighFive::File out_file(argv[2], HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);
        // HighFive::DataSet dataset = out_file.createDataSet<double>("result", HighFive::DataSpace::From(result));
        // dataset.write(result.data());
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}