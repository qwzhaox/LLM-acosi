#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <task> <data_path> <models_file> <dataset_file>"
    exit 1
fi


TASK="$1"
DATA_PATH="$2"
JSON_FILE="$DATA_PATH/$3"
DATASET_FILE="$DATA_PATH/$4"

TASK_SCRIPT=""

if [ "$TASK" = "acosi-extract" ]; then
    echo "Running ACOS extract task"
    TASK_SCRIPT="scripts/acos_extract.py"
elif [ "$TASK" = "acos-extract" ]; then
    echo "Running ACOS extract task"
    TASK_SCRIPT="scripts/acos_extract.py"
elif [ "$TASK" = "acos-extend" ]; then
    echo "Running ACOS extend task"
    TASK_SCRIPT="scripts/acos_extend.py"
else
    echo "Error: Invalid task - $TASK"
    exit 1
fi


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
        python3 "$TASK_SCRIPT" --model_name "$model" --tokenizer_name "$tokenizer" --task "$task" --dataset_file "$DATASET_FILE" --output_file "$OUTPUT_FILE" --max_new_tokens "$max_new_tokens" --remote
    else
        python3 "$TASK_SCRIPT" --model_name "$model" --tokenizer_name "$tokenizer" --task "$task" --dataset_file "$DATASET_FILE" --output_file "$OUTPUT_FILE" --max_new_tokens "$max_new_tokens"
    fi
    
done