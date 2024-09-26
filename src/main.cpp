#include <iostream>
#include "types.h"
#include "io/input/read_h5.h"
#include "array_manipulation.h"

struct FourierArray {
    Kokkos::View<int*, DEVICE> xm;
    Kokkos::View<int*, DEVICE> xn;
    Kokkos::View<double**, DEVICE> cos_coefficient;
    Kokkos::View<double**, DEVICE> sin_coefficient;

    bool has_cos;
    bool has_sin;

    int num_surfaces() const {
        if (has_sin) {
            return sin_coefficient.extent(0);
        }
        else {
            return cos_coefficient.extent(0);
        }
    }

    Kokkos::View<double***, DEVICE> evaluate(const Kokkos::View<double*, DEVICE>& u, const Kokkos::View<double*, DEVICE>& v) {
        int num_n = xn.extent(0);
        int ns = num_surfaces();

        Kokkos::View<double***, DEVICE> result("result", ns, u.extent(0), v.extent(0));

        // auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<1>>({ 0}, { ns});
        auto policy = Kokkos::RangePolicy<>(0,  ns);
        Kokkos::parallel_for("evaluate", policy, KOKKOS_LAMBDA(const int& s_idx) {
            double sum = 0.0;
            for(int u_idx = 0;u_idx < u.extent(0);u_idx++) {
                for(int v_idx = 0;v_idx < v.extent(0);v_idx++) {
                    sum = 0.0;
                    for (int m = 0; m < xm.extent(0); m++) {
                        if (has_cos) {
                            sum += cos_coefficient(s_idx, m) * Kokkos::cos(xm(m) * u(u_idx) - xn(m) * v(v_idx));
                        }
                        if (has_sin) {
                            sum += sin_coefficient(s_idx, m) * Kokkos::sin(xm(m) * u(u_idx) - xn(m) * v(v_idx));
                        }
                    }
                    result(s_idx, u_idx, v_idx) = sum;
                }
            }
            // for (int m = 0; m < xm.extent(0); m++) {
            //     if (has_cos) {
            //         sum += cos_coefficient(s_idx, m) * Kokkos::cos(xm(m) * u(u_idx) - xn(m) * v(v_idx));
            //     }
            //     if (has_sin) {
            //         sum += sin_coefficient(s_idx, m) * Kokkos::sin(xm(m) * u(u_idx) - xn(m) * v(v_idx));
            //     }
            // }
            // result(s_idx, u_idx, v_idx) = sum;
        });

        return result;
    }
};


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

    bool has_cos = file.getAttribute("has_cos").read<size_t>();
    bool has_sin = file.getAttribute("has_sin").read<size_t>();
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
    out.u = to_fixed_sized_rank<1>(H5::fetch_array<Kokkos::DynRankView<double, DEVICE>>(file, "u"));
    out.v = to_fixed_sized_rank<1>(H5::fetch_array<Kokkos::DynRankView<double, DEVICE>>(file, "v"));

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
        println("Array: {} {}", format_array(file.u), format_array(file.v));
        Kokkos::fence();
        Kokkos::Timer timer;
        auto result = file.array.evaluate(file.u, file.v);
        Kokkos::fence();
        double time = timer.seconds();
        println("Time to evaluate: {}", time);
        println("Array: {}", result(0, 0, 0));

        // HighFive::File out_file(argv[2], HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);
        // HighFive::DataSet dataset = out_file.createDataSet<double>("result", HighFive::DataSpace::From(result));
        // dataset.write(result.data());
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}