#!/bin/bash

DIRECTORY="correct/bv4"

for file in "$DIRECTORY"/*; do
    if grep -q "imeou" "$file"; then
        echo "Deleting $file"
        rm "$file"

        corresponding_sl_file="${file%.*}.sl"

        if [ -f "$corresponding_sl_file" ]; then
            echo "Deleting corresponding .sl file: $corresponding_sl_file"
            rm "$corresponding_sl_file"
        fi
    fi
done
