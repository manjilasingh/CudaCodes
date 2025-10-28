#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda

# Compile
nvcc error_saxpy.cu -o error_saxpy

# Run
./error_saxpy