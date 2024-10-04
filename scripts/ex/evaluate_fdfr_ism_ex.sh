#!/bin/bash
# the evaluate script of fdfr and ism for #2088

exec > >(cat) 2> results/evaluate_fdfr_ism_ex_error.txt

export DATA_PATH="./infer/gap_diff_per16_ex" 

export TF_ENABLE_ONEDNN_OPTS=0
time python evaluations/ex/ex_ism_fdfr.py --data_path=$DATA_PATH --method="GAP-Diff"

