#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda/12.2.1

# Compile
nvcc bandwidthtest.cu -o bandwidth_test

# Run
./bandwidth_test