#!/bin/bash

./build/bin/tests/work2_tests
./build/bin/benchmarks/work2_benchmarks --benchmark_format=json > benchmarks_results_work2.json
python ./plots/DrawPlots/complexity_chart.py \
    -j benchmarks_results_work2.json \
    --xlog --ylog -c ./plots/complexity_chart.html
python ./plots/DrawPlots/speedup_chart.py \
    -j benchmarks_results_work2.json \
    -r 'Eigen Matrix Multiplication (CPU)' \
    -t 'CUDA Matrix Multiplication (GPU)' \
    --xlog --ylog -c ./plots/speedup_chart.html