#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda

# Compile
nvcc query.cu -o query

# Run
./query