#!/bin/bash

# Default values for task, size, and model
task=""
model=""
size="N/A"
is_llama_model=false

# Function to show usage
usage() {
    echo "Usage: $0 --task TASK --model MODEL  [--size SIZE]"
    echo "  -t, --task TASK   Specify the task (acos-extract, acos-extend, acosi-extract)"
    echo "  -m, --model MODEL Specify the model (llama-2, gpt-*)"
    echo "  -s, --size SIZE   Specify the size (big, med, small)"
    exit 1
}

# Validate and update task
validate_and_update_task() {
    case $1 in
        "ae") task="acos-extract" ;;
        "ax") task="acos-extend" ;;
        "ie") task="acosi-extract" ;;
        "acos-extract"|"acos-extend"|"acosi-extract") task="$1" ;;
        *) echo "Error: Invalid task. Must be acos-extract, acos-extend, or acosi-extract."; exit 1 ;;
    esac
}

# Validate size
validate_size() {
    case $1 in
        "big") ;;
        "med") ;;
        "small") ;;
        *) echo "Error: Invalid size. Must be big, med, or small."; exit 1 ;;
    esac
}

# Validate model
validate_model() {
    case $1 in
        "llama-2") is_llama_model=true ;;
        gpt-*) is_llama_model=false ;;
        *) echo "Error: Invalid model. Must be llama-2 or start with gpt-."; exit 1 ;;
    esac
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--task) validate_and_update_task "$2"; shift ;;
        -m|--model) model="$2"; validate_model "$2"; shift ;;
        -s|--size) size="$2"; validate_size "$2"; shift ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

# Check if task and model are provided
if [ -z "$task" ] || [ -z "$model" ]; then
    echo "Error: Task and model must be provided."
    usage
fi

# Check if size is required but not provided
if [[ "$is_llama_model" == true && -z "$size" ]]; then
    echo "Error: Size is required for llama-2 model."
    usage
fi

# Output the provided values
echo "Task: $task"
echo "Size: $size"
echo "Model: $model"

if [[ "$is_llama_model" == true ]]; then
    # Run the task with the provided size and model
    if [ "$task" = "acos-extract" ]; then
        bash scripts/bash/run_task.sh "$task" "config/${size}_llama.json" "data/acos/rest/toy.txt"
        bash scripts/bash/run_task.sh "$task" "config/${size}_llama.json" "data/acos/laptop/toy.txt"
    elif [ "$task" = "acos-extend" ]; then
        # bash scripts/bash/run_task.sh "$task" "config/${size}_llama.json" "data/acos/rest/toy.txt"
        # bash scripts/bash/run_task.sh "$task" "config/${size}_llama.json" "data/acos/laptop/toy.txt"
        bash scripts/bash/run_task.sh "$task" "config/${size}_llama.json" "data/acos/shoes/toy.txt"
    elif [ "$task" = "acosi-extract" ]; then
        bash scripts/bash/run_task.sh "$task" "config/${size}_llama.json" "data/acosi/shoes/toy.txt"
    else
        echo "Error: Invalid task - $task"
        exit 1
    fi
else
    # Run the task with the provided model
    if [ "$task" = "acos-extract" ]; then
        bash scripts/bash/run_task_gpt.sh "$task" "data/acos/rest/toy.txt" "$model"
        bash scripts/bash/run_task_gpt.sh "$task" "data/acos/laptop/toy.txt" "$model"
    elif [ "$task" = "acos-extend" ]; then
        # bash scripts/bash/run_task_gpt.sh "$task" "data/acos/rest/toy.txt" "$model"
        # bash scripts/bash/run_task_gpt.sh "$task" "data/acos/laptop/toy.txt" "$model"
        bash scripts/bash/run_task_gpt.sh "$task" "data/acos/shoes/toy.txt" "$model"
    elif [ "$task" = "acosi-extract" ]; then
        bash scripts/bash/run_task_gpt.sh "$task" "data/acosi/shoes/toy.txt" "$model"
    else
        echo "Error: Invalid task - $task"
        exit 1
    fi
fi
