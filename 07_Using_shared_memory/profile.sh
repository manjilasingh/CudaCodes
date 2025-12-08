#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda

# Compile
nvcc shared-memory.cu -o shared_memory_test 

# Run
nsys profile -t cuda --stats=true ./shared_memory_test 