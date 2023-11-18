from pathlib import Path
from argparse import ArgumentParser


def get_file_path(file_name):
    # Search in the current directory and all subdirectories
    for path in Path(".").rglob(file_name):
        # Return the first match
        return path
    # Return None if no match is found
    return None


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="LLM model name")
    parser.add_argument(
        "--tokenizer_name", type=str, required=True, help="Tokenizer name"
    )
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument(
        "--remote", action="store_true", help="Whether to trust remote code"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=100, help="Max new tokens"
    )
    parser.add_argument("--dataset_file", type=str, required=True, help="Dataset file")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    args = parser.parse_args()
    return args


def remove_tags(text):
    return (
        text.replace("Aspect: ", "")
        .replace("Categroy: ", "")
        .replace("Sentiment: ", "")
        .replace("Opinion: ", "")
        .replace("Implicit/Explicit: ", "")
    )


def add_quotations(text):
    return (
        text.replace('"', "")
        .replace("'", "")
        .replace("(", "('")
        .replace(")", "')")
        .replace(", ", ",")
        .replace(",", "','")
    )


def format_output(output, response_key):
    formatted_output = []
    for out in output:
        prediction = out["generated_text"].strip()
        if response_key in prediction:
            prediction = prediction.split(response_key)[1].strip()

        prediction = remove_tags(prediction)
        prediction = add_quotations(prediction)
        prediction = prediction.lower()

        formatted_tuple = eval(prediction)
        formatted_output.append(formatted_tuple)

    return formatted_output
