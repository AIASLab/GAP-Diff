#!/bin/bash
# the evaluate script of fdfr and ism for kick-the-tires stage of #2088

exec > >(cat) 2> results/kick-the-tires/evaluate_fdfr_ism_error.txt

export DATA_PATH="./infer/kick-the-tires/gap_diff_per16" 

export TF_ENABLE_ONEDNN_OPTS=0
time python evaluations/full_ism_fdfr.py --data_path=$DATA_PATH --method="GAP-Diff"

