#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model_dir>"
    exit 1
fi

model_dir="$1"

input_directory="../Shoes-ACOSI/eval_output/$model_dir"
if [[ $model_dir == *"mvp"* ]]; then
    input_directory="$input_directory/main"
fi
input_filepath="${input_directory}/acos-extract/shoes/scores.json"

output_filepath="model_output/supervised/$model_dir/pred.json"

# Ensure input directory exists
if [ ! -d "$input_directory" ]; then
    echo "Input directory does not exist"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$(dirname "$output_filepath")"

python3 scripts/run_llm/get_pred_from_scores_file.py --input_file $input_filepath --output_file $output_filepath
