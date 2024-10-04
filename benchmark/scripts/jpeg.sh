# Do JPEG compression on protected images in #2088
export QUALITY=70

BENCHMARK_LIST=("antidb" "caat" "simac" "metacloak")

for BENCHMARK in "${BENCHMARK_LIST[@]}"; do

    export SOURCE_PATH="./benchmark/protected_images/$BENCHMARK"
    export PRE_PATH="./benchmark/protected_images/${BENCHMARK}_jpeg${QUALITY}"

    python preprocess/jpeg.py \
    --quality=$QUALITY \
    --source_path=$SOURCE_PATH \
    --output_path=$PRE_PATH
done