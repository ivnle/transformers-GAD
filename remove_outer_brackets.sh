#!/bin/bash

INPUT_DIR="correct"

# loop through all files that do not have the .sl extension
for file in $INPUT_DIR/*; do
    if [[ ! $file =~ \.sl$ ]]; then
        content=$(cat "$file" | tr -d '\n' | sed 's/^\(.\)\(.*\)\(.\)$/\2/')
        echo "$content" > "$file"
    fi
done

echo "Outer brackets removed from all applicable files in $INPUT_DIR"
