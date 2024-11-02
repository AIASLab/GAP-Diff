#!/bin/bash
# the evaluate script of fdfr and ism for #2088

exec > >(cat) 2> results/evaluate_fdfr_ism_error.txt

export DATA_PATH="./infer/gap_diff_per16" 

export TF_ENABLE_ONEDNN_OPTS=0
time python evaluations/full_ism_fdfr.py --data_path=$DATA_PATH --method="GAP-Diff"

