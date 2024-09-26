/**
 * @file types.h
 *
 * Types contains the various kokkos views used throughout the code.
*/
#pragma once
#include "kokkos.h"
#include "set_architecture.h"

#include <utility>

#ifndef GIT_COMMIT
#define GIT_COMMIT "0000000" // 0000000 means uninitialized
#endif

// Macros to work around the fact that std::max/min is not available on GPUs
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))
#define ABS(a) ((a)>0?(a):-(a))

// Memory traits
typedef Kokkos::MemoryTraits<Kokkos::RandomAccess> MemTrait_rnd;
typedef Kokkos::HostSpace Mem_HOST;

// Execution spaces
typedef Kokkos::DefaultExecutionSpace ExecSpace;
typedef Kokkos::RangePolicy<ExecSpace> RangePolicy;
typedef Kokkos::DefaultHostExecutionSpace HostExecSpace;
typedef Kokkos::RangePolicy<HostExecSpace> HostRangePolicy;

// Policies
// using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
// using Team = TeamPolicy::member_type;
// using SharedSpace = ExecSpace::scratch_memory_space;

// These arrays will be stored on the device memory
// When no device is available, they will be stored on the host memory
typedef Kokkos::View<double*, DEVICE> Array1D;
typedef Kokkos::View<const double*, DEVICE> Array1D_const;
typedef Kokkos::View<const double*, DEVICE, MemTrait_rnd> Array1D_rnd;
typedef Kokkos::View<double**, DEVICE> Array2D;
typedef Kokkos::View<double***, DEVICE> Array3D;

typedef Kokkos::View<float*, DEVICE> Array1D_f;
typedef Kokkos::View<const float*, DEVICE> Array1D_const_f;
typedef Kokkos::View<const float*, MemTrait_rnd> Array1D_rnd_f;
typedef Kokkos::View<float**, DEVICE> Array2D_f;
typedef Kokkos::View<float***, DEVICE> Array3D_f;

typedef Kokkos::View<int*, DEVICE> Array1D_i;
typedef Kokkos::View<const int*, DEVICE> Array1D_const_i;
typedef Kokkos::View<const int*, MemTrait_rnd> Array1D_rnd_i;
typedef Kokkos::View<int**, DEVICE> Array2D_i;
typedef Kokkos::View<int***, DEVICE> Array3D_i;

typedef Kokkos::View<uint64_t**, DEVICE> Array2D_u64;
typedef Kokkos::View<uint64_t*, DEVICE> Array1D_u64;

typedef Kokkos::DynRankView<double, DEVICE> NdArray_d;
typedef Kokkos::DynRankView<float, DEVICE> NdArray_f;

#define SLICE(start, end) std::pair<int, int>(start, end)

// For other system, add flags
#ifdef KOKKOS_ENABLE_CUDA
#define ENABLE_GPU
#endif

#ifdef ENABLE_GPU
typedef Kokkos::View<double*, HOST> Array1D_HOST;
typedef Kokkos::View<const double*, HOST> Array1D_HOST_const;
typedef Kokkos::View<double**, HOST> Array2D_HOST;
typedef Kokkos::View<double***, HOST> Array3D_HOST;

typedef Kokkos::View<float*, HOST> Array1D_HOST_f;
typedef Kokkos::View<const float*, HOST> Array1D_HOST_const_f;
typedef Kokkos::View<float**, HOST> Array2D_HOST_f;
typedef Kokkos::View<float***, HOST> Array3D_HOST_f;

typedef Kokkos::View<int*, HOST> Array1D_HOST_i;
typedef Kokkos::View<const int*, HOST> Array1D_HOST_const_i;
typedef Kokkos::View<int**, HOST> Array2D_HOST_i;
typedef Kokkos::View<int***, HOST> Array3D_HOST_i;

typedef Kokkos::View<uint64_t*, HOST> Array1D_HOST_u64;
typedef Kokkos::View<uint64_t**, HOST> Array2D_HOST_u64;

typedef Kokkos::DynRankView<double, HOST> NdArray_HOST_d;
typedef Kokkos::DynRankView<float, HOST> NdArray_HOST_f;
#else
typedef Array1D Array1D_HOST;
typedef Array2D Array2D_HOST;
typedef Array3D Array3D_HOST;
typedef Array1D_f Array1D_HOST_f;
typedef Array2D_f Array2D_HOST_f;
typedef Array3D_f Array3D_HOST_f;
typedef Array1D_i Array1D_HOST_i;
typedef Array2D_i Array2D_HOST_i;
typedef Array3D_i Array3D_HOST_i;
typedef Array1D_u64 Array1D_HOST_u64;
typedef Array2D_u64 Array2D_HOST_u64;
typedef Array1D Array1D_HOST_const_i;
#endif


#define NOCLASS_LAMBDA(...) [ __VA_ARGS__ ] __host__ __device__

// SIMD operations
typedef Kokkos::Experimental::native_simd<double> simd_float;

typedef Kokkos::View<size_t*> Dimensions;


#define __1D_RANGE_POLICY(N, exec) \
    Kokkos::RangePolicy<exec>(0, N)
#define __2D_RANGE_POLICY(N1, N2, exec) \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec>({ (size_t)0, (size_t)0 }, { (size_t)N1, (size_t)N2 })
#define __3D_RANGE_POLICY(N1, N2, N3, exec) \
    Kokkos::MDRangePolicy<Kokkos::Rank<3>, exec>({ (size_t)0, (size_t)0, (size_t)0 }, { (size_t)N1, (size_t)N2, (size_t)N3 })
#define __4D_RANGE_POLICY(N1, N2, N3, N4, exec) \
    Kokkos::MDRangePolicy<Kokkos::Rank<4>, exec>({ (size_t)0, (size_t)0, (size_t)0, (size_t)0 }, { (size_t)N1, (size_t)N2, (size_t)N3, (size_t)N4 })
#define __1D_RANGE_POLICY_EXT(from, to, exec) \
    Kokkos::RangePolicy<exec>((size_t)from, (size_t)to)
#define __2D_RANGE_POLICY_EXT(from1, to1, from2, to2, exec) \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec>({ (size_t)from1, (size_t)from2 }, { (size_t)to1, (size_t)to2 })
#define __3D_RANGE_POLICY_EXT(from1, to1, from2, to2, from3, to3, exec) \
    Kokkos::MDRangePolicy<Kokkos::Rank<3>, exec>({ (size_t)from1, (size_t)from2, (size_t)from3 }, { (size_t)to1, (size_t)to2, (size_t)to3 })
#define __4D_RANGE_POLICY_EXT(from1, to1, from2, to2, from3, to3, from4, to4, exec) \
    Kokkos::MDRangePolicy<Kokkos::Rank<4>, exec>({ (size_t)from1, (size_t)from2, (size_t)from3, (size_t)from4 }, { (size_t)to1, (size_t)to2, (size_t)to3, (size_t)to4 })
