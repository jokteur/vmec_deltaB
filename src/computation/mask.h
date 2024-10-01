#pragma once

#include "types.h"
#include "util/template_util.h"
#include "array_manipulation.h"

template<typename Space, typename exec = typename Kokkos::View<int*, Space>::execution_space>
size_t num_logical(const Kokkos::View<bool*, Space>& arr) {
    auto policy = Kokkos::RangePolicy<exec>(0, arr.extent(0));
    size_t num = 0;
    Kokkos::parallel_reduce("num_logical", policy, KOKKOS_LAMBDA(const size_t & i, size_t & lnum) {
        lnum += arr(i) == 1;
    }, num);
    return num;
}

template<typename Space, typename exec = typename Kokkos::View<int*, Space>::execution_space>
size_t num_logical(const Kokkos::View<int*, Space>& arr, int flag) {
    auto policy = Kokkos::RangePolicy<exec>(0, arr.extent(0));
    size_t num = 0;
    Kokkos::parallel_reduce("num_logical", policy, KOKKOS_LAMBDA(const size_t & i, size_t & lnum) {
        lnum += (bool)(arr(i) & flag);
    }, num);
    return num;
}

/**
 * @brief Masking for 1D kernels
 *
 * @tparam Space on which space the mask is stored (CPU or GPU)
 */
template<typename Space>
class Mask {
private:
    /**
     * A mask can be either a wildcard (always true),
     * a Kokkos::View<bool*, Space> comprised of only true/false values
     * or a Kokkos::View<int*, Space> comprised of flag values
     */
    Kokkos::View<bool*, Space> data_bool;
    Kokkos::View<int*, Space> data_int;
    int flag;
    bool is_wildcard_ = true;
    bool use_flag;

    friend Mask<HOST>;
    friend Mask<DEVICE>;

public:
    /**
     * @brief Default mask object (wildcard mask)
     */
    Mask() = default;
    /**
     * @brief Make a mask from a Kokkos bool view
     *
     * Data passed to this constructor is copied by reference, not by value
     * which means that the mask will be updated if the underlying data is updated
     *
     * @param data Kokkos view of bool
     */
    Mask(Kokkos::View<bool*, Space> data) : data_bool(data), is_wildcard_(false), use_flag(false) {}
    /**
     * @brief Make a mask from a Kokkos int view (flag mask)
     *
     * Data passed to this constructor is copied by reference, not by value
     * which means that the mask will be updated if the underlying data is updated
     *
     * @param data Kokkos view of int
     * @param flag flag to check for
     */
    Mask(Kokkos::View<int*, Space> data, int flag) : is_wildcard_(false), data_int(data), flag(flag), use_flag(true) {}

    void init() {
        auto policy = Kokkos::RangePolicy<typename Kokkos::View<bool*, Space>::execution_space>(0, data_bool.size());
        Kokkos::parallel_for("init mask", policy, KOKKOS_CLASS_LAMBDA(const size_t & i) {
            data_bool(i) = true;
        });
    }

    /**
     * @brief Create a new mask which is not a wildcard but all true
     *
     * @param size size of the mask
     */
    Mask(size_t size) : data_bool("mask", size), is_wildcard_(false), use_flag(false) {
        init();
    }

    /**
     * @brief Is the mask true or false at i ?
     *
     * @param i index
     * @return true or false
     */
    KOKKOS_INLINE_FUNCTION bool operator()(const size_t& i) const {
        if (use_flag) {
            return (bool)(data_int(i) & flag);
        }
        else {
            return data_bool(i);
        }
    }

    /**
     * @brief Is the mask a wildcard ?
     *
     * A wildcard indicates that the mask will always be true at any index
     *
     * @return true or false
     */
    size_t is_wildcard() const {
        return is_wildcard_;
    }

    /**
     * @brief Get a HOST copy of the mask
     *
     * @return decltype(auto)
     */
    decltype(auto) cpu_copy() const {
        Mask<HOST> mask_cpu;
        if (!is_wildcard_) {
            if (use_flag) {
                mask_cpu.data_int = to_cpu<Kokkos::View<int*, HOST>>(data_int);
            }
            else {
                mask_cpu.data_bool = to_cpu<Kokkos::View<bool*, HOST>>(data_bool);
            }
        }
        mask_cpu.is_wildcard_ = is_wildcard_;
        mask_cpu.use_flag = use_flag;
        return mask_cpu;
    }

    /**
     * @brief Size of the mask
     *
     * @param default_size if the mask is a wildcard, return this size
     * @return size_t number of true elements in the mask
     */
    size_t size(size_t default_size) const {
        if (is_wildcard_) {
            return default_size;
        }
        else {
            if (use_flag) {
                return num_logical(data_int, flag);
            }
            else {
                return num_logical(data_bool);
            }
        }
    }
};
