#!/bin/bash

# Define arguments
CACHE_DIR="/nobackup2/yf/mila/GD_caches"
BASE_GRAMMAR_DIR="/nobackup2/yf/mila/GD/examples/grammars/"
#GRAMMAR_FILE="string_01.ebnf"
#GRAMMAR_FILES=("string_01.ebnf" "string_0.ebnf" "string_1.ebnf")
GRAMMAR_FILES=("string_start_w_1_all_0_len_3.ebnf")
#NUM_BEAMS=5
REPETITION_PENALTIES=(1.0)
#REPETITION_PENALTIES=(1)
STRING_LENGTHS=(3)
TOP_PS=(1.0)
#TEMPERATURES=(1.1 0.8 1.5 0.6 2 5)
TEMPERATURES=(1.0)
ITER=500

#PROMPT="Generate a random binary string of length ${STRING_LENGTHS}?"

# Models to test
#models=("meta-llama/Llama-2-7b-hf"
#"meta-llama/Llama-2-13b-hf"
#"meta-llama/Llama-2-70b-hf"
#"mistralai/Mixtral-8x7B-Instruct-v0.1")

#models=("meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf" "meta-llama/Llama-2-70b-chat-hf")
#models=("meta-llama/Llama-2-13b-hf")
#models=("meta-llama/Llama-2-7b-hf")
models=("mistralai/Mistral-7B-Instruct-v0.1")

# Iteration
for MODEL_ID in "${models[@]}"; do
    for GRAMMAR_FILE in "${GRAMMAR_FILES[@]}"; do
        for STRING_LENGTH in "${STRING_LENGTHS[@]}"; do
            for REPETITION_PENALTY in "${REPETITION_PENALTIES[@]}"; do
                for TEMPERATURE in "${TEMPERATURES[@]}"; do
                    for TOP_P in "${TOP_PS[@]}"; do  # Assuming you want to loop over TOP_PS
                        PROMPT="Be a helpful assistant. Generate a random binary string of length ${STRING_LENGTH}? Directly show the generated string without explanation."
                        CUDA_VISIBLE_DEVICES=1 python run_inference_GCD.py \
                            --model_id "$MODEL_ID" \
                            --cache_dir "$CACHE_DIR" \
                            --base_grammar_dir "$BASE_GRAMMAR_DIR" \
                            --grammar_file "$GRAMMAR_FILE" \
                            --num_return_sequences 1 \
                            --repetition_penalty $REPETITION_PENALTY \
                            --string_length $STRING_LENGTH \
                            --prompt "$PROMPT" \
                            --top_p $TOP_P \
                            --temperature $TEMPERATURE \
                            --iter $ITER \
                            --do_sample \
                            --log_file '/nobackup2/yf/mila/GD/log/log_mistral_7b_gcd.log'\
                            --max_new_tokens 4
                    done  # top_p
                done  # temperature
            done  # repetition penalty
        done  # string length
    done  # grammar
done  # model

