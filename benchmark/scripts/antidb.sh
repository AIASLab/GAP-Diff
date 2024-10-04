#!/bin/bash

exec > >(cat) 2> benchmark/results/antidb_error.txt

ID_LIST=("n000061" "n000089" "n000090" "n000154" "n000161")
for ID in "${ID_LIST[@]}"; do
    export MODEL_PATH=$(realpath "/root/autodl-tmp/stable-diffusion/stable-diffusion-2-1-base")
    export CLEAN_TRAIN_DIR="data/test_dataset/$ID/set_A" 
    export CLEAN_ADV_DIR="data/test_dataset/$ID/set_B"
    export OUTPUT_DIR="benchmark/protected_images/antidb/$ID"
    export CLASS_DIR="data/class-person"

    # ------------------------- Train ASPL on set B -------------------------
    mkdir -p $OUTPUT_DIR

    time accelerate launch benchmark/antidb.py \
    --pretrained_model_name_or_path=$MODEL_PATH  \
    --enable_xformers_memory_efficient_attention \
    --instance_data_dir_for_train=$CLEAN_TRAIN_DIR \
    --instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
    --instance_prompt="a photo of sks person" \
    --class_data_dir=$CLASS_DIR \
    --num_class_images=200 \
    --class_prompt="a photo of person" \
    --output_dir=$OUTPUT_DIR \
    --center_crop \
    --with_prior_preservation \
    --prior_loss_weight=1.0 \
    --resolution=512 \
    --train_text_encoder \
    --train_batch_size=1 \
    --max_train_steps=50 \
    --max_f_train_steps=3 \
    --max_adv_train_steps=6 \
    --checkpointing_iterations=50 \
    --learning_rate=5e-7 \
    --pgd_alpha=5e-3 \
    --pgd_eps=0.0627 
    
done