#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda/12.2.1

# Compile
nvcc coalescing.cu -o coalescing_test 

# Run
./coalescing_test 