#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <ABSA_TASK> <dataset_file> <model or models_file>"
    exit 1
fi

ABSA_TASK="$1"
DATASET_FILE="$2"
PYTHON_SCRIPT="scripts/run_llm/main.py"

max_length=0
max_new_tokens=0

if [ "$ABSA_TASK" = "acosi-extract" ]; then
    echo "Running ACOSI extract"
    max_length=4096
    max_new_tokens=1024
elif [ "$ABSA_TASK" = "acos-extract" ]; then
    echo "Running ACOS extract"
    max_length=4096
    max_new_tokens=1024
elif [ "$ABSA_TASK" = "acos-extend" ]; then
    echo "Running ACOS extend"
    max_length=4096
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

# check whether the third argument is a file or a model name
if [ -f "$3" ]; then
    LLM_FILE="$3"

    # Check if the JSON file exists
    if [ ! -f "$LLM_FILE" ]; then
        echo "Error: File not found - $JSON_FILE"
        exit 1
    fi

    # Loop through the JSON array
    jq -c '.[]' "$LLM_FILE" | while read -r line; do
        # Extract fields from JSON
        model=$(echo "$line" | jq -r '.model')
        tokenizer=$(echo "$line" | jq -r '.tokenizer')
        task=$(echo "$line" | jq -r '.task')
        remote=$(echo "$line" | jq -r '.remote')
        k_examples=$(echo "$line" | jq -r '.k_examples')
        selection_method=$(echo "$line" | jq -r '.selection_method')

        model_type=$(echo "$model" | cut -d'/' -f1)
        model_name=$(echo "$model" | cut -d'/' -f2-)

        # Set the output file name
        OUTPUT_FILE="model_output/${model_type}-${selection_method}-${k_examples}/${model_name}/${ABSA_TASK}/${DATASET}/output"

        # Set the remote flag for the Python script
        if [ "$remote" = "true" ]; then
            python3 "$PYTHON_SCRIPT" --model_name "$model" --tokenizer_name "$tokenizer" --task "$task" --absa_task "$ABSA_TASK" --dataset_file "$DATASET_FILE" --output_file "$OUTPUT_FILE" --max_new_tokens "$max_new_tokens" --max_length "$max_length" --k_examples "$k_examples" --selection_method "$selection_method" --remote --is_combo_prompt
        else
            python3 "$PYTHON_SCRIPT" --model_name "$model" --tokenizer_name "$tokenizer" --task "$task" --absa_task "$ABSA_TASK" --dataset_file "$DATASET_FILE" --output_file "$OUTPUT_FILE" --max_new_tokens "$max_new_tokens" --max_length "$max_length" --k_examples "$k_examples" --selection_method "$selection_method" --is_combo_prompt
        fi
    done
else
    MODEL="$3"
    for k_examples in 5 10; do
        for selection_method in random tf-idf; do
            OUTPUT_FILE="model_output/gpt-${selection_method}-${k_examples}/${MODEL}/${ABSA_TASK}/${DATASET}/output"
            python3 "$PYTHON_SCRIPT" --model_name "$MODEL" --absa_task "$ABSA_TASK" --dataset_file "$DATASET_FILE" --output_file "$OUTPUT_FILE" --max_new_tokens "$max_new_tokens" --max_length "$max_length" --k_examples "$k_examples" --selection_method "$selection_method"
        done
    done
fi
