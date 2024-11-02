#!/bin/bash
# the evaluate script of brisque and serfiq for #2088

exec > >(cat) 2> results/evaluate_brisque_serfiq_error.txt

export DATA_PATH="./infer/gap_diff_per16"


export MXNET_USE_FUSION=0
export MXNET_CUDNN_LIB_CHECKING=0   
time python evaluations/full_brisque.py --data_path=$DATA_PATH --method="GAP-Diff"
time python evaluations/full_ser.py --data_path=$DATA_PATH --method="GAP-Diff"

