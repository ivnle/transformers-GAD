#!/bin/bash

INPUT_DIR="correct"
OUTPUT_DIR="prompts/SLIA"

mkdir -p $OUTPUT_DIR

for file in $INPUT_DIR/*.sl; do
    filename=$(basename -- "$file")
    basename="${filename%.sl}"

    prefix=$(echo $basename | sed -E 's/([-_][a-zA-Z0-9_]+)$//')

    # Find matching files and select three at random
    matches=($(find $INPUT_DIR -name "$prefix*.sl" | shuf -n 3))

    # Check if there are enough matches, if not select any 3 files randomly
    if [ ${#matches[@]} -lt 3 ]; then
        matches=($(find $INPUT_DIR -name "*.sl" | shuf -n 3))
    fi

    python generate_icl_prompts.py "$basename" "${matches[@]}" > "$OUTPUT_DIR/$basename.txt"
done

echo "Prompts have been generated in $OUTPUT_DIR"
