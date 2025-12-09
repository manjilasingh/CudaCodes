#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda/12.2.1

# Compile
nvcc query.cu -o query

# Run
./query