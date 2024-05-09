#!/bin/bash

ROOT_DIR="prompts/bv4"

find "$ROOT_DIR" -name '*.txt' -type f | while read file; do
    if grep -q 'imeou' "$file"; then
        echo "Found 'imeou' in: $file"
    fi
done
