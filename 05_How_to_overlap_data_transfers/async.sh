#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda

# Compile
nvcc async.cu -o async

# Run
./async
./async
./async
./async
./async
./async
./async
./async
./async
./async