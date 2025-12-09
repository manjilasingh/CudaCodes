#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda/12.2.1

# Compile
nvcc error_saxpy.cu -o error_saxpy

# Run
./error_saxpy