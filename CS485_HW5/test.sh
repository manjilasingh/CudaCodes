#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda/12.2.1

# Compile with optimizations
nvcc -O3 hw_base.cu -o base

OUTPUT_DIR="/scratch/$USER/base"
mkdir -p $OUTPUT_DIR

echo "Starting Game of Life Performance Tests"
echo "========================================"

# Test 1: Grid size scaling (fixed iterations)
echo "Test 1: Grid Size Scaling"
for size in 5000; do
    echo "Running ${size}x${size} grid, 1000 iterations..."
    nsys profile -t cuda --stats=true -o ${OUTPUT_DIR}/profile_size_${size} \
        ./base $size 1000 ${OUTPUT_DIR}/size_${size}_iter_1000.txt
done

#