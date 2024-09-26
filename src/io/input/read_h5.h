#pragma once

#include "io/h5.h"
#include "kokkos.h"
#include "types.h"
#include "io/output/print.h"

namespace H5 {
    template<typename View>
    View fetch_array(HighFive::File& file, const std::string& dataset_name, bool transpose = false) {
        HighFive::DataSet dataset = file.getDataSet(dataset_name);
        HighFive::DataSpace dataspace = dataset.getSpace();
        std::vector<size_t> dims = dataspace.getDimensions();
        if (dims.size() == 1) {
            /* From HighFive to Kokkos */
            View array(dataset_name, dims[0]);
            // If compiled with GPU, this array lives on the GPU, thus we need a host view
            auto host_array = Kokkos::create_mirror_view(array);

            std::vector<typename View::value_type> values;
            dataset.read(values);

            for (size_t i = 0; i < dims[0]; ++i) {
                host_array(i) = values[i];
            }
            Kokkos::deep_copy(array, host_array); // This does nothing if host == device
            return array;
        }
        else if (dims.size() == 2) {
            /* From HighFive to Kokkos */
            std::vector<std::vector<typename View::value_type>> values;
            dataset.read(values);
            if (transpose) {
                View array(dataset_name, dims[1], dims[0]);
                auto host_array = Kokkos::create_mirror_view(array);
                for (size_t i = 0; i < dims[0]; ++i) {
                    for (size_t j = 0; j < dims[1]; ++j) {
                        host_array(j, i) = values[i][j];
                    }
                }
                Kokkos::deep_copy(array, host_array);
                return array;
            }
            else {
                View array(dataset_name, dims[0], dims[1]);
                auto host_array = Kokkos::create_mirror_view(array);
                for (size_t i = 0; i < dims[0]; ++i) {
                    for (size_t j = 0; j < dims[1]; ++j) {
                        host_array(i, j) = values[i][j];
                    }
                }
                Kokkos::deep_copy(array, host_array);
                return array;
            }
        }
        else if (dims.size() == 3) {
            std::vector<std::vector<std::vector<typename View::value_type>>> values;
            dataset.read(values);
            if (transpose) {
                View array(dataset_name, dims[2], dims[1], dims[0]);
                auto host_array = Kokkos::create_mirror_view(array);
                for (size_t i = 0; i < dims[0]; ++i) {
                    for (size_t j = 0; j < dims[1]; ++j) {
                        for (size_t k = 0; k < dims[2]; ++k) {
                            host_array(k, j, i) = values[i][j][k];
                        }
                    }
                }
                Kokkos::deep_copy(array, host_array);
                return array;
            }
            else {
                View array(dataset_name, dims[0], dims[1], dims[2]);
                auto host_array = Kokkos::create_mirror_view(array);
                for (size_t i = 0; i < dims[0]; ++i) {
                    for (size_t j = 0; j < dims[1]; ++j) {
                        for (size_t k = 0; k < dims[2]; ++k) {
                            host_array(i, j, k) = values[i][j][k];
                        }
                    }
                }
                Kokkos::deep_copy(array, host_array);
                return array;
            }
        }
        else if (dims.size() == 4) {
            std::vector<std::vector<std::vector<std::vector<typename View::value_type>>>> values;
            dataset.read(values);
            if (transpose) {
                View array(dataset_name, dims[3], dims[2], dims[1], dims[0]);
                auto host_array = Kokkos::create_mirror_view(array);
                for (size_t i = 0; i < dims[0]; ++i) {
                    for (size_t j = 0; j < dims[1]; ++j) {
                        for (size_t k = 0; k < dims[2]; ++k) {
                            for (size_t l = 0; l < dims[3]; ++l) {
                                host_array(l, k, j, i) = values[i][j][k][l];
                            }
                        }
                    }
                }
                Kokkos::deep_copy(array, host_array);
                return array;
            }
            else {
                View array(dataset_name, dims[0], dims[1], dims[2], dims[3]);
                auto host_array = Kokkos::create_mirror_view(array);
                for (size_t i = 0; i < dims[0]; ++i) {
                    for (size_t j = 0; j < dims[1]; ++j) {
                        for (size_t k = 0; k < dims[2]; ++k) {
                            for (size_t l = 0; l < dims[3]; ++l) {
                                host_array(i, j, k, l) = values[i][j][k][l];
                            }
                        }
                    }
                }
                Kokkos::deep_copy(array, host_array);
                return array;
            }
        }
        else {
            throw std::runtime_error("h5 reader: Higher dimensional array currently not supported");
        }
    }
}
