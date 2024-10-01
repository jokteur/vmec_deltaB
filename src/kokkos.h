#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_DynRankView.hpp>
#include <Kokkos_Random.hpp>


#define HOST Kokkos::HostSpace
#define HOSTexec Kokkos::HostSpace::execution_space
#define HOSTmem Kokkos::HostSpace::memory_space
#define HOSTlayout Kokkos::HostSpace::array_layout


#ifdef KOKKOS_ENABLE_CUDA
#define MemorySpace Kokkos::CudaSpace
#define DEVICE Kokkos::CudaSpace
#define DEVICEexec Kokkos::Cuda
#define DEVICEmem Kokkos::CudaSpace::memory_space
#define DEVICElayout Kokkos::Cuda::array_layout
#endif

// No support for other vendors yet
// #ifdef KOKKOS_ENABLE_HIP
// #define MemorySpace Kokkos::Experimental::HIPSpace
// #endif
// #ifdef KOKKOS_ENABLE_OPENMPTARGET
// #define MemSpace Kokkos::OpenMPTargetSpace
// #endif

#ifndef DEVICE
#define MemorySpace Kokkos::HostSpace
#define DEVICE Kokkos::HostSpace
#define DEVICEexec Kokkos::View<double*, HOST>::execution_space
#define DEVICEmem Kokkos::HostSpace::memory_space
#define DEVICElayout Kokkos::OpenMP::array_layout
#endif

