#pragma once

#include "io/h5.h"
#include <vector>
#include <string>
#include "types.h"
#include "io/output/print.h"
#include "util/template_util.h"
#include "computation/array_manipulation.h"

namespace H5 {
    template<typename Arr>
    inline auto get_local_dimensions(const Arr& arr) {
        if constexpr (is_kokkos_view<Arr>()) {
            std::vector<size_t> dims(arr.rank());
            for (size_t i = 0;i < arr.rank();i++) {
                dims[i] = arr.extent(i);
            }
            return dims;
        }
        else {
            return arr.shape();
        }
    }
    template<typename Arr>
    inline auto get_global_dimensions(const Arr& arr) {
        if constexpr (is_kokkos_view<Arr>()) {
            std::vector<size_t> dims(arr.rank());
            for (size_t i = 0;i < arr.rank();i++) {
                size_t local_size = arr.extent(i);
                size_t total_size;
                MPI_Allreduce(&local_size, &total_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
                dims[i] = total_size;
            }
            return dims;
        }
        else {
            throw std::runtime_error("Error: get_global_dimensions not implemented for non-kokkos views");
        }
    }


    template<typename File, typename FileOrGroup, typename Array>
    inline void write_dataset(File& file, FileOrGroup& group, const std::string& dataset_name, const Array& data, HighFive::DataTransferProps xfer_props = HighFive::DataTransferProps(), int compression = 4) {
        using namespace HighFive;
        int mpi_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

        if (compression < 0 || compression > 9)
            throw std::runtime_error("Error: compression level must be between 0 and 9");

        if constexpr (is_kokkos_view<Array>()) {
            using T = typename Array::value_type;
            using Space = typename Array::memory_space;
            if constexpr (std::is_same_v<Space, Kokkos::HostSpace>) {
                auto shape = get_local_dimensions(data);
                auto dcpl = HighFive::DataSetCreateProps{};
                if (compression) {
                    std::vector<hsize_t> chunks(shape.begin(), shape.end());
                    for (size_t i = 0;i < shape.size();i++) {
                        // 32 KB chunk size
                        chunks[i] = std::min(shape[i], (size_t)4096);
                    }
                    dcpl.add(HighFive::Shuffle());
                    dcpl.add(HighFive::Chunking(chunks));
                    dcpl.add(HighFive::Deflate(compression));
                }

                DataSet dataset = group.template createDataSet<T>(dataset_name, DataSpace(shape), dcpl);
                if (data.rank() == 1) {
                    auto data_ = (T*)data.data();
                    dataset.write(data_, xfer_props);
                }
                else if (data.rank() == 2) {
                    auto data_ = (T**)data.data();
                    dataset.write(data_, xfer_props);
                }
                else if (data.rank() == 3) {
                    auto data_ = (T***)data.data();
                    dataset.write(data_, xfer_props);
                }
                else if (data.rank() == 4) {
                    auto data_ = (T****)data.data();
                    dataset.write(data_, xfer_props);
                }
                else if (data.rank() == 5) {
                    auto data_ = (T*****)data.data();
                    dataset.write(data_, xfer_props);
                }
                else {
                    throw std::runtime_error("Error: rank > 4 not supported");
                }
            }
            else {
                auto array = to_cpu<Kokkos::View<typename Array::data_type, HOST>>(data);
                write_dataset(file, group, dataset_name, array, xfer_props, compression);
            }
        }
        else {
            if (compression)
                throw std::runtime_error("Error: compression not supported for non-kokkos views");
            group.createDataSet(dataset_name, data);
        }
        // Ensure that everything has been written do disk.
        file.flush();
    }
}