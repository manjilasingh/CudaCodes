#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda

# Compile
nvcc profile.cu -o profile_test

# Run
nsys profile -t cuda --stats=true ./profile_test