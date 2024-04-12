#!/bin/bash

GPUs=(0)
gpu_counter=0

# Define default values for the arguments
#MODEL_ID="bigcode/starcoder2-15b"
#MODEL_IDS=("codellama/CodeLlama-7b-Instruct-hf" "codellama/CodeLlama-13b-Instruct-hf" "codellama/CodeLlama-34b-Instruct-hf" "codellama/CodeLlama-70b-Instruct-hf")
MODEL_IDS=("codellama/CodeLlama-7b-Instruct-hf")
ITER=50
MAX_NEW_TOKENS=512
PROMPT_TYPES=("bare")
# GRAMMAR_DIR="/nobackup2/yf/lily/GD/examples/grammars/"
GRAMMAR_DIR="/nobackup2/yf/lily/GD/examples/sygus/"
# GRAMMAR_FILE="string_start_w_1_all_0_len_3.ebnf"
GRAMMAR_PROMPT_FILES=(
"/nobackup2/yf/lily/GD/benchmarks/comp/2019/General_Track/bv-conditional-inverses/find_inv_bvsge_bvadd_4bit.sl"
"/nobackup2/yf/lily/GD/benchmarks/comp/2019/General_Track/woosuk/sygus_iter_26_0.sl"
"/nobackup2/yf/lily/GD/benchmarks/comp/2019/General_Track/from_2018/CrCi/CrCy_1-P5-D5-sIn1.sl"
)

# Call the Python script with the defined arguments
for MODEL_ID in "${MODEL_IDS[@]}"; do
    for PROMPT_TYPE in "${PROMPT_TYPES[@]}"; do
        for GRAMMAR_PROMPT_FILE in "${GRAMMAR_PROMPT_FILES[@]}"; do
            gpu=${GPUs[$gpu_counter]}
            echo "Running model: $MODEL_ID, on GPU: $gpu"
            CUDA_VISIBLE_DEVICES=$gpu python run_inference_gcd_build_oracle_trie.py \
            --model_id "$MODEL_ID" \
            --cache_dir "/nobackup2/yf/lily/GD_caches/" \
            --num_return_sequences 1 \
            --repetition_penalty 1.0 \
            --iter $ITER \
            --temperature 1.0 \
            --top_p 1.0 \
            --top_k 0 \
            --max_new_tokens $MAX_NEW_TOKENS \
            --prompt_type "$PROMPT_TYPE" \
            --output_folder "/nobackup2/yf/lily/GD/results/" \
            --base_grammar_dir "$GRAMMAR_DIR" \
            --instruct_prompt_file "/nobackup2/yf/lily/GD/prompts/pre_prompt.jsonl" \
            --dtype "float32" \
            --grammar_prompt_file "$GRAMMAR_PROMPT_FILE" \
            --device "cpu" &
            let "gpu_counter = (gpu_counter + 1) % ${#GPUs[@]}"
        done
    done
done

wait

echo "All experiments have finished."

# --grammar_file "$GRAMMAR_FILE" \