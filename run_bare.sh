#!/bin/bash

GPUs=(0 1 2 3 4)
gpu_counter=0

# Define default values for the arguments
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.1"
ITER=100
MAX_NEW_TOKENS=512
SYGUS_PROMPT_FILE="/nobackup2/yf/mila/GD/prompts/pre_prompt.jsonl"
PROMPT_TYPES=("bare" "completion")

# Call the Python script with the defined arguments
for PROMPT_TYPE in "${PROMPT_TYPES[@]}"; do
    gpu=${GPUs[$gpu_counter]}
    echo "Running model: $MODEL_ID, on GPU: $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python run_inference_bare.py \
    --model_id "$MODEL_ID" \
    --cache_dir "/nobackup2/yf/mila/GD_caches/" \
    --num_return_sequences 1 \
    --repetition_penalty 1.0 \
    --iter $ITER \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k 0 \
    --max_new_tokens $MAX_NEW_TOKENS \
    --sygus_prompt_file "$SYGUS_PROMPT_FILE" \
    --prompt_type "$PROMPT_TYPE" \
    --output_folder "/nobackup2/yf/mila/GD/results/" &
    let "gpu_counter = (gpu_counter + 1) % ${#GPUs[@]}"
done

wait

echo "All experiments have finished."