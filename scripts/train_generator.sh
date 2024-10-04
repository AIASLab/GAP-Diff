#!/bin/bash
# the training script for the generator of #2088
#exec > >(cat) 2> results/train_generator_error.txt

export MODEL_PATH=$(realpath "/root/autodl-tmp/stable-diffusion/stable-diffusion-2-1-base")
export OUTPUT_DIR="weights/model/"

mkdir -p $OUTPUT_DIR

accelerate launch train_generator.py \
  --pretrained_model_name_or_path="$MODEL_PATH"  \
  --enable_xformers_memory_efficient_attention \
  --instance_data_dir="data/FFHQ512_2W" \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks person" \
  --resolution=512 \
  --max_train_steps=50 \
  --train_batch_size=6 \
  --training
