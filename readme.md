# Installation

After you cloned the repository, you must download the submodules:
```
cd vmec_deltaB
git submodule update --init --recursive
```

Now create a build folder:
```
mkdir build
cd build
```

To compile with parallel CPU, you can do:
```
cmake .. -DKokkos_ENABLE_OPENMP=ON -DCMAKE_BUILD_TYPE=Release
```

If you have an Nvidia GPU, you can do:
```
cmake .. -DKokkos_ENABLE_OPENMP=ON -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON -DCMAKE_CXX_COMPILER=/path/to/vmec_deltaB/bin/nvcc_wrapper
```

And finally, to compile:
```
make -j
```
