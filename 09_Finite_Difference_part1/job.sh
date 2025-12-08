#!/bin/bash
echo "Running on: $(hostname)"

module purge
module load cuda/12.2.1

echo "===== compiling ====="
nvcc finite-difference.cu -o fd_test

echo "===== running finite difference test ====="
./fd_test
