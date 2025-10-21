#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda

# Compile
nvcc add.cu -o add

# Run
nsys profile -t cuda --stats=true ./add
