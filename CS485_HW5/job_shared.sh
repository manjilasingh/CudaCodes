#!/bin/bash

cd $PBS_O_WORKDIR
module load cuda

# Compile with optimizations
nvcc -O3 hw.shared.cu -o shared

OUTPUT_DIR="/scratch/$USER/shared"
mkdir -p $OUTPUT_DIR

echo "Starting Game of Life Performance Tests"
echo "========================================"

# Test 1: Grid size scaling (fixed iterations)
echo "Test 1: Grid Size Scaling"
for size in 500 1000 5000 10000 15000 20000; do
    echo "Running ${size}x${size} grid, 1000 iterations..."
    nsys profile -t cuda --stats=true -o ${OUTPUT_DIR}/profile_size_${size} \
        ./shared $size 1000 ${OUTPUT_DIR}/size_${size}_iter_1000.txt
done

# Test 2: Iteration scaling (fixed grid)
echo "Test 2: Iteration Scaling"
for iters in 100 500 1000 2500 5000; do
    echo "Running 10000x10000 grid, ${iters} iterations..."
    nsys profile -t cuda --stats=true -o ${OUTPUT_DIR}/profile_iter_${iters} \
        ./shared 10000 $iters ${OUTPUT_DIR}/size_10000_iter_${iters}.txt
done

echo "Tests complete! Results in $OUTPUT_DIR"