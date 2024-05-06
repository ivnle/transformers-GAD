#!/bin/bash

# Directory containing the files
DIRECTORY="correct/bv4"

# Verify if the directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Directory does not exist: $DIRECTORY"
    exit 1
fi

# Loop through all files in the directory and modify them
find "$DIRECTORY" -type f -print | while read file; do
    echo "Processing file: $file"
    sed -i 's/_ //g' "$file"
done

echo "All files have been processed."
