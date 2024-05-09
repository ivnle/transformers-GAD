#!/bin/bash

SOURCE_DIR="prompts/bv4"
DEST_DIR="prompts/bv4_no_grammar"

mkdir -p "$DEST_DIR"

for input_file in "$SOURCE_DIR"/*.txt; do
    filename=$(basename "$input_file")

    output_file="$DEST_DIR/${filename%.txt}.txt"

    python3 delete_grammar_prompt.py "$input_file" "$output_file"

    echo "Processed $input_file -> $output_file"
done