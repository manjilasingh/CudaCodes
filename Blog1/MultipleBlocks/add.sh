#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda

# Compile
nvcc add.cu -o add

# Run
./add
