#!/bin/bash

./build/bin/tests/work5_tests
./build/bin/benchmarks/work5_benchmarks --benchmark_format=json > benchmarks_results_work5.json
python ./plots/DrawPlots/complexity_chart.py \
    -j benchmarks_results_work5.json \
    --xlog --ylog -c ./plots/complexity_chart.html
python ./plots/DrawPlots/speedup_chart.py \
    -j benchmarks_results_work5.json \
    -r 'CUDA Shmem Reduction Sum (GPU)' \
    -t 'CUDA Warp Shuffle Reduction Sum (GPU)' \
    --xlog --ylog -c ./plots/speedup_chart.html