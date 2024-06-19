#!/bin/bash
module purge
module load gcc
export HDF5_DIR=/scratch365/njamieso/hdf5
export PATH=$HDF5_DIR/bin:$PATH
export LD_LIBRARY_PATH=$HDF5_DIR/lib:$LD_LIBRARY_PATH

# Build executable
rm -f tree
g++ -c tree.cpp -fopenmp -I${HDF5_DIR}/include -o tree.o
g++ -fopenmp tree.o -Wl,-rpath,${HDF5_DIR}/lib -L${HDF5_DIR}/lib -lhdf5 -lz -o tree
