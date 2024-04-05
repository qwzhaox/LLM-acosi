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

        # Construct the command using an array
        cmd=(
            python3 "$PYTHON_SCRIPT"
            --model_name "$model"
            --tokenizer_name "$tokenizer"
            --task "$task"
            --max_new_tokens "$max_new_tokens"
            --max_length "$max_length"
            --absa_task "$ABSA_TASK"
            --dataset_file "$DATASET_FILE"
            --k_examples "$k_examples"
            --selection_method "$selection_method"
            --is_combo_prompt
        )

        if [ "$remote" = "true" ]; then
            cmd+=(--remote)
        fi

        cmd0=("${cmd[@]}")

        # comment this if running a variation
        # OUTPUT_FILE="model_output/${model_type}-${selection_method}-${k_examples}/${model_name}/${ABSA_TASK}/${DATASET}/output"

        # uncomment this to run the limited train set variation
        # if [ $DATASET = "shoes" ]; then
        #     continue
        # fi
        OUTPUT_FILE="model_output/${model_type}-${selection_method}-${k_examples}-906/${model_name}/${ABSA_TASK}/${DATASET}/output"
        cmd0+=(--limit 906)

        # uncomment this to run the [acos-extend alt annotation source] variation
        # if [ $DATASET != "shoes" ] || [ $ABSA_TASK != "acos-extend" ]; then
        #     continue
        # fi

        # uncomment this to run the [acos-extend alt annotation source: mvp] variation
        # OUTPUT_FILE="model_output/${model_type}-${selection_method}-${k_examples}-mvp-seed-5/${model_name}/${ABSA_TASK}/${DATASET}/output"
        # cmd0+=(--annotation_source mvp-seed-5)

        # uncomment this to run the [acos-extend alt annotation source: gen-scl-nat] variation
        # OUTPUT_FILE="model_output/${model_type}-${selection_method}-${k_examples}-gen-scl-nat/${model_name}/${ABSA_TASK}/${DATASET}/output"
        # cmd0+=(--annotation_source gen-scl-nat)

        # Run the command
        cmd0+=(--output_file $OUTPUT_FILE)
        "${cmd0[@]}"

        # uncomment this to run the [acos-extend alt annotation source: mvp] variation for all the seeds
        # for seed in 10 15 20 25; do
        #     cmd0=("${cmd[@]}")
        #     OUTPUT_FILE="model_output/${model_type}-${selection_method}-${k_examples}-mvp-seed-$seed/${model_name}/${ABSA_TASK}/${DATASET}/output"
        #     cmd0+=(--annotation_source mvp-seed-$seed)

        #     cmd0+=(--output_file $OUTPUT_FILE)
        #     "${cmd0[@]}"
        # done
    done
else
    MODEL="$3"
    for k_examples in 5 10; do
        for selection_method in random tf-idf; do
            # Construct the command using an array
            cmd=(
                python3 "$PYTHON_SCRIPT"
                --model_name "$MODEL"
                --max_new_tokens "$max_new_tokens"
                --max_length "$max_length"
                --absa_task "$ABSA_TASK"
                --dataset_file "$DATASET_FILE"
                --k_examples "$k_examples"
                --selection_method "$selection_method"
            )

            cmd0=("${cmd[@]}")

            # comment this if running a variation
            # OUTPUT_FILE="model_output/gpt-${selection_method}-${k_examples}/${MODEL}/${ABSA_TASK}/${DATASET}/output"

            # uncomment this to run the limited train set variation
            # if [ $DATASET = "shoes" ]; then
            #     continue
            # fi
            OUTPUT_FILE="model_output/gpt-${selection_method}-${k_examples}-906/${MODEL}/${ABSA_TASK}/${DATASET}/output"
            cmd0+=(--limit 906)

            # uncomment this to run the [acos-extend alt annotation source] variation
            # if [ $DATASET != "shoes" ] || [ $ABSA_TASK != "acos-extend" ]; then
            #     continue
            # fi

            # uncomment this to run the [acos-extend alt annotation source: mvp] variation
            # OUTPUT_FILE="model_output/gpt-${selection_method}-${k_examples}-mvp-seed-5/${MODEL}/${ABSA_TASK}/${DATASET}/output"
            # cmd0+=(--annotation_source mvp-seed-5)

            # uncomment this to run the [acos-extend alt annotation source: gen-scl-nat] variation
            # OUTPUT_FILE="model_output/gpt-${selection_method}-${k_examples}-gen-scl-nat/${MODEL}/${ABSA_TASK}/${DATASET}/output"
            # cmd0+=(--annotation_source gen-scl-nat)

            cmd0+=(--output_file $OUTPUT_FILE)
            "${cmd0[@]}"

            # uncomment this to run the [acos-extend alt annotation source: mvp] variation for all the seeds
            # for seed in 10 15 20 25; do
            #     cmd0=("${cmd[@]}")
            #     OUTPUT_FILE="model_output/gpt-${selection_method}-${k_examples}-mvp-seed-$seed/${MODEL}/${ABSA_TASK}/${DATASET}/output"
            #     cmd0+=(--annotation_source mvp-seed-$seed)

            #     cmd0+=(--output_file $OUTPUT_FILE)
            #     "${cmd0[@]}"
            # done
        done
    done
fi
