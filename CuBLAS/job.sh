#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda

# Compile
nvcc -lcublas matmul.cu -o matmul

./matmul

