#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda

# Compile
nvcc saxpy.cu -o saxpy

# Run
./saxpy