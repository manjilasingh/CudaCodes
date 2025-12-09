#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda/12.2.1

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