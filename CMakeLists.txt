cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_STANDARD 17)


project(vmec_deltaB)

# ------
# Kokkos
# ------
set(Kokkos_ENABLE_OPENMP ON)
add_subdirectory(external/kokkos)


# --------------------------
# Format (advanced printing)
# --------------------------
if (NOT TARGET fmt)
    add_subdirectory(external/fmt)
endif()

# ---
# MPI
# ---
find_package(MPI)

# --------
# HighFive
# --------
# set(HIGHFIVE_PARALLEL_HDF5 ON)
set(HIGHFIVE_USE_BOOST OFF)
add_subdirectory(external/HighFive)



include_directories(src)
file(GLOB source_list RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    src/*.cpp
    src/io/input/*.cpp
    src/io/output/*.cpp
    src/util/*.cpp
)

add_executable(${PROJECT_NAME} ${source_list})
target_link_libraries(${PROJECT_NAME} kokkos HighFive fmt::fmt stdc++)
target_include_directories(${PROJECT_NAME} PUBLIC kokkos)