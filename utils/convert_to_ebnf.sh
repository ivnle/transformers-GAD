#!/bin/bash

INPUT_DIR="benchmarks/comp/2019/General_Track/bv-conditional-inverses"
OUTPUT_DIR="examples/sygus/bv4"

mkdir -p $OUTPUT_DIR

for file in $INPUT_DIR/*.sl; do
    filename=$(basename -- "$file")
    basename="${filename%.*}"
    output_file="$OUTPUT_DIR/$basename.ebnf"
    python tools/to_ebnf.py -s 1 "$file" > "$output_file"
done

echo "Conversion complete. EBNF files saved in $OUTPUT_DIR"
