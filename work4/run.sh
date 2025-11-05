#!/bin/bash

./build/bin/tests/work4_tests
./build/bin/benchmarks/work4_benchmarks --benchmark_format=json > benchmarks_results_work4.json
python ./plots/DrawPlots/complexity_chart.py \
    -j benchmarks_results_work4.json \
    --xlog --ylog -c ./plots/complexity_chart.html
python ./plots/DrawPlots/speedup_chart.py \
    -j benchmarks_results_work4.json \
    -r 'CUDA Shmem Matrix Multiplication (GPU)' \
    -t 'CUDA WMMA Matrix Multiplication (GPU)' \
    --xlog --ylog -c ./plots/speedup_chart.html