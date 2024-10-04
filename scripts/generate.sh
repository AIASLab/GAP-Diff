#!/bin/bash
# the generate script for #2088

exec > >(cat) 2> results/generate_error.txt

ID_LIST=("n000061" "n000089" "n000090" "n000154" "n000161")
for ID in "${ID_LIST[@]}"; do
    time python generate.py \
    --generator_path="weights/model/G_per16_pretrain.pth" \
    --source_path="data/test_dataset/$ID/set_B" \
    --save_path="protected_images/gap_diff_per16/$ID" \
    --noise_budget="16.0"
done
