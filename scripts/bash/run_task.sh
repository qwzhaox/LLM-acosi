#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <ABSA_TASK> <models_file> <dataset_file>"
    exit 1
fi


ABSA_TASK="$1"
LLM_FILE="$2"
DATASET_FILE="$3"

ABSA_TASK_SCRIPT=""

if [ "$ABSA_TASK" = "acosi-extract" ]; then
    echo "Running ACOS extract ABSA_TASK"
    ABSA_TASK_SCRIPT="scripts/acos_extract.py"
elif [ "$ABSA_TASK" = "acos-extract" ]; then
    echo "Running ACOS extract ABSA_TASK"
    ABSA_TASK_SCRIPT="scripts/acos_extract.py"
elif [ "$ABSA_TASK" = "acos-extend" ]; then
    echo "Running ACOS extend ABSA_TASK"
    ABSA_TASK_SCRIPT="scripts/acos_extend.py"
else
    echo "Error: Invalid ABSA_TASK - $ABSA_TASK"
    exit 1
fi


# Check if the JSON file exists
if [ ! -f "$LLM_FILE" ]; then
    echo "Error: File not found - $JSON_FILE"
    exit 1
fi

# Check if the dataset file exists
if [ ! -f "$DATASET_FILE" ]; then
    echo "Error: Dataset file not found - $DATASET_FILE"
    exit 1
fi


# Loop through the JSON array
jq -c '.[]' "$LLM_FILE" | while read -r line; do
    # Extract fields from JSON
    model=$(echo "$line" | jq -r '.model')
    tokenizer=$(echo "$line" | jq -r '.tokenizer')
    base=$(echo "$line" | jq -r '.base')
    base_tokenizer=$(echo "$line" | jq -r '.base_tokenizer')
    task=$(echo "$line" | jq -r '.task')
    max_new_tokens=$(echo "$line" | jq -r '.max_new_tokens')
    remote=$(echo "$line" | jq -r '.remote')

    # Set the output file name
    OUTPUT_FILE="data/model_output/${model}_${ABSA_TASK}_output.pkl"

    # Set the remote flag for the Python script
    if [ "$remote" = "true" ]; then
        python3 "$ABSA_TASK_SCRIPT" --model_name "$model" --tokenizer_name "$tokenizer" --task "$task" --dataset_file "$DATASET_FILE" --output_file "$OUTPUT_FILE" --max_new_tokens "$max_new_tokens" --remote
    else
        python3 "$ABSA_TASK_SCRIPT" --model_name "$model" --tokenizer_name "$tokenizer" --task "$task" --dataset_file "$DATASET_FILE" --output_file "$OUTPUT_FILE" --max_new_tokens "$max_new_tokens"
    fi
    
done