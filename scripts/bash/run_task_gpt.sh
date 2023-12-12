#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <ABSA_TASK> <dataset_file>"
    exit 1
fi


ABSA_TASK="$1"
DATASET_FILE="$2"

python_file=""
max_new_tokens=512
max_length=1024
model="gpt-4-short"

if [ "$ABSA_TASK" = "acosi-extract" ]; then
    echo "Running ACOSI extract"
    max_length=2048
    max_new_tokens=1024
    python_file="scripts/acosi_extract.py"
elif [ "$ABSA_TASK" = "acos-extract" ]; then
    echo "Running ACOS extract"
    max_length=2048
    max_new_tokens=1024
    python_file="scripts/acos_extract.py"
elif [ "$ABSA_TASK" = "acos-extend" ]; then
    echo "Running ACOS extend"
    max_length=1024
    max_new_tokens=512
    python_file="scripts/acos_extend.py"
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

OUTPUT_FILE="data/model_output/${model}/${ABSA_TASK}/${DATASET}/output.pkl"

# Run script
python3 "$python_file" --model_name "$model" --dataset_file "$DATASET_FILE" --output_file "$OUTPUT_FILE" --max_new_tokens "$max_new_tokens" --max_length "$max_length"
