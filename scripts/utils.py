from pathlib import Path
from argparse import ArgumentParser
from itertools import chain


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
        .replace("Category: ", "")
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


def format_output(output, response_key, response_head):
    output = list(chain(*output))
    formatted_output = []
    for out in output:
        prediction = out["generated_text"].strip()

        if response_key in prediction:
            prediction = prediction.split(response_key)[1].strip()
        if response_head in prediction:
            prediction = prediction.split(response_head)[1].strip()

        if "[" in prediction and "]" in prediction:
            prediction = prediction[prediction.find("[") : prediction.find("]") + 1]
        elif "[" in prediction:
            prediction = (
                prediction[prediction.find("[") : prediction.rfind(")") + 1] + "]"
            )
        else:
            formatted_output.append([])
            continue

        prediction = remove_tags(prediction)
        prediction = add_quotations(prediction)
        prediction = prediction.lower()

        try:
            formatted_tuple = eval(prediction)
            formatted_output.append(formatted_tuple)
        except Exception as e:
            print(e)
            print("SKIPPING ANNOTATION")
            formatted_output.append([])

    return formatted_output
