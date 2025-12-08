#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda

# Compile
nvcc coalescing.cu -o coalescing_test 

# Run
./coalescing_test 