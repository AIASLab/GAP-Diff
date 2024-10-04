#!/bin/bash
# the evaluate script of fdfr and ism for #2088

exec > >(cat) 2> benchmark/results/evaluate_fdfr_ism_error.txt


BENCHMARK_LIST=("antidb" "caat" "simac" "metacloak")

for BENCHMARK in "${BENCHMARK_LIST[@]}"; do
    export DATA_PATH="./benchmark/infer/${BENCHMARK}_jpeg70" 

    export TF_ENABLE_ONEDNN_OPTS=0
    time python evaluations/full_ism_fdfr.py --data_path=$DATA_PATH --method=$BENCHMARK
done

