#include "random.h"

size_t random_seed(size_t init_seed) {
    static size_t counter = 12;
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (init_seed == SIZE_MAX) {
        return counter++ * (mpi_rank + 1);
    }
    else {
        return init_seed + counter++ * (mpi_rank + 1);
    }
}