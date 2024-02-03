#!/bin/bash

# Define arguments
CACHE_DIR="/nobackup2/yf/mila/GD_caches"
BASE_GRAMMAR_DIR="/nobackup2/yf/mila/GD/examples/grammars/"
#GRAMMAR_FILE="string_01.ebnf"
GRAMMAR_FILES=("string_01.ebnf" "string_0.ebnf" "string_1.ebnf")
NUM_BEAMS=3
REPETITION_PENALTIES=(1 1.5 2 5 10 20)
#REPETITION_PENALTIES=(1)
STRING_LENGTHS=(2 3 4 5 6 7)
#STRING_LENGTHS=(2)
TOP_K=1000
TOP_PS=(0.9 0.8 0.7 0.95)
TEMPERATURES=(1.5 2 1.0 5 10 100)
#TEMPERATURES=(1)
ITER=100

PROMPT="Generate a random binary string of length ${STRING_LENGTH}?"

# Models to test
#models=("meta-llama/Llama-2-7b-hf"
#"meta-llama/Llama-2-13b-hf"
#"meta-llama/Llama-2-70b-hf"
#"mistralai/Mixtral-8x7B-Instruct-v0.1")

models=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "meta-llama/Llama-2-70b-hf")

# Iteration
for MODEL_ID in "${models[@]}"; do
    for GRAMMAR_FILE in "${GRAMMAR_FILES[@]}"; do
        for STRING_LENGTH in "${STRING_LENGTHS[@]}"; do
            for REPETITION_PENALTY in "${REPETITION_PENALTIES[@]}"; do
                for TEMPERATURE in "${TEMPERATURES[@]}"; do
                    for TOP_P in "${TOP_PS[@]}"; do  # Assuming you want to loop over TOP_PS
                        PROMPT="Randomly generate a binary string of length at most ${STRING_LENGTH}?"
                        python run_inference_grammar_constrained.py \
                            --model_id "$MODEL_ID" \
                            --cache_dir "$CACHE_DIR" \
                            --base_grammar_dir "$BASE_GRAMMAR_DIR" \
                            --grammar_file "$GRAMMAR_FILE" \
                            --num_return_sequences 1 \
                            --max_length 50 \
                            --seed 42 \
                            --num_beams $NUM_BEAMS \
                            --repetition_penalty $REPETITION_PENALTY \
                            --string_length $STRING_LENGTH \
                            --prompt "$PROMPT" \
                            --top_k $TOP_K \
                            --top_p $TOP_P \
                            --temperature $TEMPERATURE \
                            --iter $ITER \
                            --do_sample
                        # The above line is the last argument, so no backslash is needed
                    done  # Corresponds to for TOP_P in "${TOP_PS[@]}"
                done  # Corresponds to for TEMPERATURE in "${TEMPERATURES[@]}"
            done  # Corresponds to for REPETITION_PENALTY in "${REPETITION_PENALTIES[@]}"
        done  # Corresponds to for STRING_LENGTH in "${STRING_LENGTHS[@]}"
    done  # Corresponds to for GRAMMAR_FILE in "${GRAMMAR_FILES[@]}"
done  # Corresponds to for MODEL_ID in "${models[@]}"

