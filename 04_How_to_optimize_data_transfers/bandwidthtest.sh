#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda

# Compile
nvcc bandwidthtest.cu -o bandwidth_test

# Run
./bandwidth_test