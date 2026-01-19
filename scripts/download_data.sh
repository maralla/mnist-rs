#!/bin/bash

set -e

DATA_DIR="${1:-./data}"
BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"

FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

echo "MNIST Dataset Downloader"
echo "========================"
echo "Data directory: $DATA_DIR"
echo

mkdir -p "$DATA_DIR"

for file in "${FILES[@]}"; do
    output_file="${file%.gz}"
    output_path="$DATA_DIR/$output_file"

    if [ -f "$output_path" ]; then
        echo "✓ $output_file already exists, skipping"
        continue
    fi

    echo "Downloading $file..."
    curl -sS -o "$DATA_DIR/$file" "$BASE_URL/$file"

    echo "Extracting $file..."
    gunzip -f "$DATA_DIR/$file"

    echo "✓ $output_file ready"
done

echo
echo "Download complete!"
echo "Files in $DATA_DIR:"
ls -lh "$DATA_DIR"
