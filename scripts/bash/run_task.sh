#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <ABSA_TASK> <models_file> <dataset_file>"
    exit 1
fi


ABSA_TASK="$1"
LLM_FILE="$2"
DATASET_FILE="$3"

ABSA_TASK_SCRIPT=""
max_new_tokens=512
max_length=1024

if [ "$ABSA_TASK" = "acosi-extract" ]; then
    echo "Running ACOSI extract"
    ABSA_TASK_SCRIPT="scripts/acos_extract.py"
    max_length=2048
    max_new_tokens=1024
elif [ "$ABSA_TASK" = "acos-extract" ]; then
    echo "Running ACOS extract"
    ABSA_TASK_SCRIPT="scripts/acos_extract.py"
    max_length=2048
    max_new_tokens=1024
elif [ "$ABSA_TASK" = "acos-extend" ]; then
    echo "Running ACOS extend"
    ABSA_TASK_SCRIPT="scripts/acos_extend.py"
    max_length=1024
    max_new_tokens=512
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

DATASET=""

if [[ "$DATASET_FILE" == *"rest"* ]]; then
    DATASET="rest"
elif [[ "$DATASET_FILE" == *"laptop"* ]]; then
    DATASET="laptop"
elif [[ "$DATASET_FILE" == *"shoes"* ]]; then
    DATASET="shoes"
else
    echo "Error: Invalid dataset file - $DATASET_FILE"
    exit 1
fi

# Loop through the JSON array
jq -c '.[]' "$LLM_FILE" | while read -r line; do
    # Extract fields from JSON
    model=$(echo "$line" | jq -r '.model')
    tokenizer=$(echo "$line" | jq -r '.tokenizer')
    task=$(echo "$line" | jq -r '.task')
    remote=$(echo "$line" | jq -r '.remote')

    # Set the output file name
    OUTPUT_FILE="data/model_output/${model}/${ABSA_TASK}/${DATASET}/output.pkl"

    # Set the remote flag for the Python script
    if [ "$remote" = "true" ]; then
        python3 "$ABSA_TASK_SCRIPT" --model_name "$model" --tokenizer_name "$tokenizer" --task "$task" --dataset_file "$DATASET_FILE" --output_file "$OUTPUT_FILE" --max_new_tokens "$max_new_tokens" --max_length "$max_length" --remote
    else
        python3 "$ABSA_TASK_SCRIPT" --model_name "$model" --tokenizer_name "$tokenizer" --task "$task" --dataset_file "$DATASET_FILE" --output_file "$OUTPUT_FILE" --max_new_tokens "$max_new_tokens" --max_length "$max_length" 
    fi
    
done