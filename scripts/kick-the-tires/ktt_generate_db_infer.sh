#!/bin/bash
# the generate, dreambooth training and inference script for kick-the-tires stage of #2088

exec > >(cat) 2> results/kick-the-tires/generate_db_infer_error.txt

ID_LIST=("n000089")
for ID in "${ID_LIST[@]}"; do
    python generate.py \
    --generator_path="weights/model/G_per16_pretrain.pth" \
    --source_path="data/test_dataset/$ID/set_B" \
    --save_path="protected_images/kick-the-tires/gap_diff_per16/$ID" \
    --noise_budget="16.0"

    # ------------------------- Train DreamBooth on perturbed examples -------------------------
    export MODEL_PATH=$(realpath "/root/autodl-tmp/stable-diffusion/stable-diffusion-2-1-base")
    export CLASS_DIR="data/class-person-2-1"  
    export INSTANCE_DIR="./protected_images/kick-the-tires/gap_diff_per16/$ID"
    export DREAMBOOTH_OUTPUT_DIR="weights/dreambooth/DREAMBOOTH_$ID"

    accelerate launch train_dreambooth.py \
        --gradient_checkpointing \
        --pretrained_model_name_or_path=$MODEL_PATH  \
        --enable_xformers_memory_efficient_attention \
        --train_text_encoder \
        --instance_data_dir=$INSTANCE_DIR \
        --class_data_dir=$CLASS_DIR \
        --output_dir=$DREAMBOOTH_OUTPUT_DIR \
        --with_prior_preservation \
        --prior_loss_weight=1.0 \
        --instance_prompt="a photo of sks person" \
        --class_prompt="a photo of person" \
        --inference_prompt="a photo of sks person;a dslr portrait of sks person" \
        --resolution=512 \
        --train_batch_size=2 \
        --gradient_accumulation_steps=1 \
        --learning_rate=5e-7 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=200 \
        --max_train_steps=1000 \
        --checkpointing_steps=1000 \
        --center_crop \
        --mixed_precision=bf16 \
        --prior_generation_precision=bf16 \
        --sample_batch_size=16

    # ------------------------- Using customized DreamBooth to infer-------------------------
    export INFER_PATH="$DREAMBOOTH_OUTPUT_DIR/checkpoint-1000"
    export INFER_OUTPATH="./infer/kick-the-tires/gap_diff_per16/$ID"

    python infer.py \
        --prompts="a photo of sks person" \
        --model_path=$INFER_PATH \
        --output_dir=$INFER_OUTPATH

    rm -rf "$DREAMBOOTH_OUTPUT_DIR"
done
  
