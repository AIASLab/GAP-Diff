#!/bin/bash

exec > >(cat) 2> benchmark/results/caat_error.txt

ID_LIST=("n000061" "n000089" "n000090" "n000154" "n000161")

for ID in "${ID_LIST[@]}"; do

  export MODEL_PATH=$(realpath "/root/autodl-tmp/stable-diffusion/stable-diffusion-2-1-base")
  export OUTPUT_DIR="benchmark/protected_images/caat/$ID"
  export INSTANCE_DIR="data/test_dataset/$ID/set_B"

  time accelerate launch benchmark/caat.py \
    --pretrained_model_name_or_path=$MODEL_PATH \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --instance_prompt="a photo of a person" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --lr_warmup_steps=0 \
    --max_train_steps=250 \
    --hflip \
    --mixed_precision bf16  \
    --alpha=5e-3  \
    --eps=0.0627450980392157
done
