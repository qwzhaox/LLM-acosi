#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <data_path> <models_file> <dataset_file>"
    exit 1
fi

# Assign arguments to variables
DATA_PATH="$1"
JSON_FILE="$DATA_PATH/$2"
DATASET_FILE="$DATA_PATH/$3"

# Check if the data path exists
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Data path not found - $DATA_PATH"
    exit 1
fi

# Check if the JSON file exists
if [ ! -f "$JSON_FILE" ]; then
    echo "Error: File not found - $JSON_FILE"
    exit 1
fi

# Check if the dataset file exists
if [ ! -f "$DATASET_FILE" ]; then
    echo "Error: Dataset file not found - $DATASET_FILE"
    exit 1
fi

# Loop through the JSON array
jq -c '.[]' "$JSON_FILE" | while read -r line; do
    # Extract fields from JSON
    model=$(echo "$line" | jq -r '.model')
    tokenizer=$(echo "$line" | jq -r '.tokenizer')
    base=$(echo "$line" | jq -r '.base')
    base_tokenizer=$(echo "$line" | jq -r '.base_tokenizer')
    task=$(echo "$line" | jq -r '.task')
    max_new_tokens=$(echo "$line" | jq -r '.max_new_tokens')
    remote=$(echo "$line" | jq -r '.remote')

    # Set the output file name
    OUTPUT_FILE="$DATA_PATH/${model}_output.json"

    # Set the remote flag for the Python script
    if [ "$remote" = "true" ]; then
        python3 src/llm/pipeline.py -m "$model" -t "$tokenizer" -b "$base" -bt "$base_tokenizer" -a "$task" -d "$DATASET_FILE" -o "$OUTPUT_FILE" -tok "$max_new_tokens" -r
    else
        python3 src/llm/pipeline.py -m "$model" -t "$tokenizer" -b "$base" -bt "$base_tokenizer" -a "$task" -d "$DATASET_FILE" -o "$OUTPUT_FILE" -tok "$max_new_tokens"
    fi
    
done