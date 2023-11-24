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


def flatten_output(output):
    if type(output[0]) is list:
        return list(chain(*output))
    return output


def remove_tags(text):
    return (
        text.replace("Aspect: ", "")
        .replace("Category: ", "")
        .replace("Sentiment: ", "")
        .replace("Implicit Opinion: ", "")
        .replace("Opinion: ", "")
        .replace("Implicit/Explicit: ", "")
    )


def add_quotations(text, response_head):
    text = (
        text.replace('"', "")
        .replace("'", "")
        .replace("(", "('")
        .replace(")", "')")
        .replace(", ", ",")
        .replace(",", "','")
    )
    if response_head == "Opinion spans:":
        text = text.replace("[", "['").replace("]", "']")
    return text


def clean_output(out, response_key, response_head):
    prediction = out["generated_text"].strip()

    if response_key in prediction:
        prediction = prediction.split(response_key)[1].strip()
    if response_head in prediction:
        prediction = prediction.split(response_head)[1].strip()

    return prediction


def extract_list_str(prediction):
    if "[" in prediction and "]" in prediction:
        prediction = prediction[prediction.find("[") : prediction.find("]") + 1]
    elif "[" in prediction:
        prediction = prediction[prediction.find("[") : prediction.rfind(")") + 1] + "]"
    else:
        return "", False
    return prediction, True


def format_list_str(prediction, response_head):
    prediction = remove_tags(prediction)
    prediction = add_quotations(prediction, response_head)
    prediction = prediction.lower()
    return prediction


def eval_list_str(prediction):
    try:
        formatted_tuples = eval(prediction)
    except Exception as e:
        print(e)
        print("Invalid output: ", prediction, "\n")
        return []
    return formatted_tuples


def format_output(output, response_key, response_head):
    output = flatten_output(output)
    formatted_output = []
    for out in output:
        prediction = clean_output(out, response_key, response_head)
        prediction, valid = extract_list_str(prediction)
        if not valid:
            formatted_output.append([])
            continue

        prediction = format_list_str(prediction, response_head)
        formatted_tuples = eval_list_str(prediction)
        formatted_output.append(formatted_tuples)

    return formatted_output
