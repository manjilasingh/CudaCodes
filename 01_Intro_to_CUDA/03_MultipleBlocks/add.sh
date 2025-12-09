#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda/12.2.1

# Compile
nvcc add.cu -o add

# Run
./add
