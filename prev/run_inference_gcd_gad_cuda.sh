#!/bin/bash

GPUs=(7)
gpu_counter=0

# Define the threshold for free memory (in MB) to consider a GPU as idle
FREE_MEMORY_THRESHOLD=40000

# Split MODEL_ID into an array
#MODEL_IDS=("bigcode/starcoder2-7b" "bigcode/starcoder2-3b")
MODEL_IDS=("bigcode/starcoder2-15b")
ITER=100
MAX_NEW_TOKENS=512
#PROMPT_TYPES=("bare" "completion")
PROMPT_TYPES=("bare")

check_gpu_idle() {
    local gpu_id=$1
    free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu_id | awk '{print $1}')
    if [[ $free_memory -gt $FREE_MEMORY_THRESHOLD ]]; then
        return 0 # GPU is idle
    else
        return 1 # GPU is not idle
    fi
}

launch_experiment() {
    local gpu_id=$1
    local model_id=$2
    local prompt_type=$3
    echo "Running model: $model_id, on GPU: $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id python run_inference_gcd_gad_cuda.py \
        --model_id "$model_id" \
        --cache_dir "/nobackup2/yf/mila/GD_caches/" \
        --num_return_sequences 1 \
        --repetition_penalty 1.0 \
        --iter $ITER \
        --temperature 1.0 \
        --top_p 1.0 \
        --top_k 0 \
        --max_new_tokens $MAX_NEW_TOKENS \
        --sygus_prompt_file "/nobackup2/yf/mila/GD/prompts/pre_prompt.jsonl" \
        --prompt_type "$prompt_type" \
        --output_folder "/nobackup2/yf/mila/GD/results/" \
        --base_grammar_dir "/nobackup2/yf/mila/GD/examples/sygus/" &
}

# Main loop
for model_id in "${MODEL_IDS[@]}"; do
    for prompt_type in "${PROMPT_TYPES[@]}"; do
        gpu_found=false
        while ! $gpu_found; do
            for gpu in "${GPUs[@]}"; do
                if check_gpu_idle $gpu; then
                    launch_experiment $gpu "$model_id" "$prompt_type"
                    gpu_found=true
                    break 2 # Exit both loops
                fi
            done
            echo "No idle GPU found, waiting..."
            sleep 60 # Wait for 60 seconds before checking again
        done
    done
done

wait

echo "All experiments have finished."
