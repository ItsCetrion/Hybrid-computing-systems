#!/bin/bash

./build/bin/tests/work1_tests
./build/bin/benchmarks/work1_benchmarks --benchmark_format=json --benchmark_out=./plots/benchmarks_data/results.json > /dev/null
python3 ./plots/main.py