#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <ABSA_TASK> <dataset_file> <output_file>"
    exit 1
fi


ABSA_TASK="$1"
DATASET_FILE="$2"
OUTPUT_FILE="$3"

max_new_tokens=512
max_length=1024

if [ "$ABSA_TASK" = "acosi-extract" ]; then
    echo "Running ACOSI extract"
    max_length=2048
    max_new_tokens=1024
elif [ "$ABSA_TASK" = "acos-extract" ]; then
    echo "Running ACOS extract"
    max_length=2048
    max_new_tokens=1024
elif [ "$ABSA_TASK" = "acos-extend" ]; then
    echo "Running ACOS extend"
    max_length=1024
    max_new_tokens=512
else
    echo "Error: Invalid ABSA_TASK - $ABSA_TASK"
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

# Run script
python3 "scripts/gpt4.py" --absa_task "$ABSA_TASK" --dataset_file "$DATASET_FILE" --output_file "$OUTPUT_FILE"
