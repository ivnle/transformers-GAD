#!/bin/bash

GPUs=(0)
gpu_counter=0

# Define default values for the arguments
MODEL_IDS=("mistralai/Mistral-7B-Instruct-v0.2")
ITER=50
MAX_NEW_TOKENS=512
CACHE_DIR="/path/to/where/you/store/hf/models"
OUTPUT_FOLDER="results/SLIA"
TEST_FOLDER="correct/"
PROMPT_FOLDER="prompts/SLIA"

# Outer loop for MODEL_ID
for MODEL_ID in "${MODEL_IDS[@]}"; do
    gpu=${GPUs[$gpu_counter]}
    echo "Running model: $MODEL_ID, on GPU: $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python run_inference_bare.py \
    --model_id "$MODEL_ID" \
    --cache_dir "$CACHE_DIR" \
    --num_return_sequences 1 \
    --repetition_penalty 1.0 \
    --iter $ITER \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k 0 \
    --max_new_tokens $MAX_NEW_TOKENS \
    --dtype "float32" \
    --output_folder "$OUTPUT_FOLDER" \
    --test_folder "$TEST_FOLDER" \
    --prompt_folder "$PROMPT_FOLDER" \
    --seed 42 \
    --device "cuda" &
    let "gpu_counter = (gpu_counter + 1) % ${#GPUs[@]}"
done
wait

echo "All experiments have finished."