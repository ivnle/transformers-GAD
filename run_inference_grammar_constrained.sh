#!/bin/bash

# Define arguments
CACHE_DIR="/nobackup2/yf/mila/GD_caches"
BASE_GRAMMAR_DIR="/nobackup2/yf/mila/GD/examples/grammars/"
#GRAMMAR_FILE="string_01.ebnf"
#GRAMMAR_FILES=("string_01.ebnf" "string_0.ebnf" "string_1.ebnf")
GRAMMAR_FILES=("string_start_w_1_all_0.ebnf")
NUM_BEAMS=5
REPETITION_PENALTIES=(1 1.5 2 5)
#REPETITION_PENALTIES=(1)
STRING_LENGTHS=(5)
TOP_PS=(0.95)
TEMPERATURES=(1.1 0.8 1.5 0.6 2 5)
#TEMPERATURES=(1)
ITER=1

#PROMPT="Generate a random binary string of length ${STRING_LENGTHS}?"

# Models to test
#models=("meta-llama/Llama-2-7b-hf"
#"meta-llama/Llama-2-13b-hf"
#"meta-llama/Llama-2-70b-hf"
#"mistralai/Mixtral-8x7B-Instruct-v0.1")

models=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "meta-llama/Llama-2-70b-hf")
#models=("meta-llama/Llama-2-13b-hf")

# Iteration
for MODEL_ID in "${models[@]}"; do
    for GRAMMAR_FILE in "${GRAMMAR_FILES[@]}"; do
        for STRING_LENGTH in "${STRING_LENGTHS[@]}"; do
            for REPETITION_PENALTY in "${REPETITION_PENALTIES[@]}"; do
                for TEMPERATURE in "${TEMPERATURES[@]}"; do
                    for TOP_P in "${TOP_PS[@]}"; do  # Assuming you want to loop over TOP_PS
                        PROMPT="Generate a random binary string of length ${STRING_LENGTH}?"
                        python run_inference_grammar_constrained.py \
                            --model_id "$MODEL_ID" \
                            --cache_dir "$CACHE_DIR" \
                            --base_grammar_dir "$BASE_GRAMMAR_DIR" \
                            --grammar_file "$GRAMMAR_FILE" \
                            --num_return_sequences 1 \
                            --max_length 50 \
                            --num_beams $NUM_BEAMS \
                            --repetition_penalty $REPETITION_PENALTY \
                            --string_length $STRING_LENGTH \
                            --prompt "$PROMPT" \
                            --top_p $TOP_P \
                            --temperature $TEMPERATURE \
                            --iter $ITER \
                            --do_sample \
                            --log_file '/nobackup2/yf/mila/GD/log/log_one_time_greedy.txt'\
                            --max_new_tokens 20
                    done  # top_p
                done  # temperature
            done  # repetition penalty
        done  # string length
    done  # grammar
done  # model

