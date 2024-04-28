#!/bin/bash

GPUs=(0 1 2 3 4 5 6 7)
gpu_counter=0

# Define default values for the arguments
MODEL_IDS=("mistralai/Mistral-7B-Instruct-v0.2")
ITER=10
MAX_NEW_TOKENS=512
CACHE_DIR="/path/to/where/you/store/hf/models"
OUTPUT_FOLDER="results/SLIA"
TEST_FOLDER="correct/"
PROMPT_FOLDER="prompts/SLIA"
TRIE_FOLDER="tries/SLIA"

is_gpu_free() {
    # If nvidia-smi query is successful and the GPU is idle (no process found), return 0 (success)
    [ -z "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader,nounits)" ]
}

# Call the Python script with the defined arguments
for MODEL_ID in "${MODEL_IDS[@]}"; do
    while true; do
        gpu=${GPUs[$gpu_counter]}
        if is_gpu_free $gpu; then
            echo "GPU $gpu is free. Running model: $model_path, on GPU: $gpu"
            CUDA_VISIBLE_DEVICES=$gpu python run_inference_gcd_build_oracle_trie.py \
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
            --trie_folder "$TRIE_FOLDER" \
            --seed 42 \
            --device "cuda" &
            let "gpu_counter = (gpu_counter + 1) % ${#GPUs[@]}"
            break
        else
            echo "GPU $gpu is busy. Waiting for 60 seconds..."
            sleep 60
        fi
    done
done

wait
echo "All experiments have finished."

# --grammar_file "$GRAMMAR_FILE" \