# Do JPEG compression on protected images in #2088
export QUALITY=70
export SOURCE_PATH="./protected_images/gap_diff_per16"
export PRE_PATH="./protected_images/gap_diff_per16_jpeg$QUALITY"

python preprocess/jpeg.py \
  --quality=$QUALITY \
  --source_path=$SOURCE_PATH \
  --output_path=$PRE_PATH