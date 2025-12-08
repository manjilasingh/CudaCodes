#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda

# Compile
nvcc transpose.cu -o transpose

# Run
nsys profile -t cuda --stats=true ./transpose 