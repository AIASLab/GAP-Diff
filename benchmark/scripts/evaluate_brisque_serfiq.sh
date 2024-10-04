#!/bin/bash
# the evaluate script of brisque and serfiq for #2088

exec > >(cat) 2> benchmark/results/evaluate_brisque_serfiq_error.txt

BENCHMARK_LIST=("antidb" "caat" "simac" "metacloak")

for BENCHMARK in "${BENCHMARK_LIST[@]}"; do
    export DATA_PATH="./benchmark/infer/${BENCHMARK}_jpeg70"

    export MXNET_USE_FUSION=0
    export MXNET_CUDNN_LIB_CHECKING=0 
    time python evaluations/full_brisque.py --data_path=$DATA_PATH --method=$BENCHMARK  
    time python evaluations/full_ser.py --data_path=$DATA_PATH --method=$BENCHMARK

done

