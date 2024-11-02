#!/bin/bash
#
#ID_LIST=("n000061" "n000088" "n000090" "n000154" "n000161")
ID_LIST=("n000061" "n000089" "n000090" "n000154" "n000161")
for ID in "${ID_LIST[@]}"; do
    export MODEL_PATH=$(realpath "/root/autodl-tmp/stable-diffusion/stable-diffusion-2-1-base")
    export OUTPUT_DIR="./outputs/$ID"
    export INSTANCE_DIR="./data/gap_diff_per16/$ID"
    export CLASS_DIR="data/class-person"

    accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_PATH  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="a photo of sks person" \
    --class_prompt="a photo of person" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-3 \
    --learning_rate_1d=1e-6 \
    --train_text_encoder \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --max_train_steps=500

    export INFER_OUTPUT=$(realpath "/root/GAP-Diff/infer/svdiff/$ID")

    python infer.py \
        --model_path=$OUTPUT_DIR \
        --output_dir=$INFER_OUTPUT \
        --diffusion_path=$MODEL_PATH
    
    rm -rf "$OUTPUT_DIR"
done
