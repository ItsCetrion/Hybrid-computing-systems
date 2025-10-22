#!/bin/bash

./build/bin/tests/work3_tests
./build/bin/benchmarks/work3_benchmarks --benchmark_format=json > benchmarks_results_work3.json
python ./plots/DrawPlots/complexity_chart.py \
    -j benchmarks_results_work3.json \
    --xlog --ylog -c ./plots/complexity_chart.html
python ./plots/DrawPlots/speedup_chart.py \
    -j benchmarks_results_work3.json \
    -r 'CUDA Native Matrix Multiplication (GPU)' \
    -t 'CUDA Shmem Matrix Multiplication (GPU)' \
    --xlog --ylog -c ./plots/speedup_chart.html