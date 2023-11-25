import re
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from itertools import chain
from pickle import dump
from string import punctuation


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


def clean_output(out, response_key, response_head):
    prediction = out["generated_text"].strip()

    if response_key in prediction:
        prediction = prediction.split(response_key)[1].strip()
    if response_head in prediction:
        prediction = prediction.split(response_head)[1].strip()

    return prediction


# def extract_list_str(prediction):
#     if "[" in prediction and "]" in prediction:
#         prediction = prediction[prediction.find("[") : prediction.find("]") + 1]
#     elif "[" in prediction:
#         prediction = prediction[prediction.find("[") : prediction.rfind(")") + 1] + "]"
#     else:
#         return "", False
#     return prediction, True


# def format_list_str(prediction, response_head):
#     if "span" in response_head.lower():
#         for i, char in prediction:

#     else:

#     prediction = prediction.lower()
#     return prediction


# def eval_list_str(prediction):
#     try:
#         formatted_tuples = literal_eval(prediction)
#     except Exception as e:
#         print("ERROR MESSAGE:", e)
#         print("INVALID OUTPUT: ", prediction, "\n")
#         return []
#     return formatted_tuples


def clean_punctuation(words):
    punc = re.compile(f"[{re.escape(punctuation)}]")
    words = punc.sub(" \\g<0> ", words)

    # remove extra spaces
    words = words.strip()
    words = " ".join(words.split())
    return words


def extract_spans(seq):
    quints = []
    sents = [s.strip() for s in seq.split("[SSEP]")]
    for s in sents:
        try:
            tok_list = ["[C]", "[S]", "[A]", "[O]", "[I]"]

            for tok in tok_list:
                if tok not in s:
                    s += " {} null".format(tok)
            index_ac = s.index("[C]")
            index_sp = s.index("[S]")
            index_at = s.index("[A]")
            index_ot = s.index("[O]")
            index_ie = s.index("[I]")

            combined_list = [index_ac, index_sp, index_at, index_ot, index_ie]
            arg_index_list = list(np.argsort(combined_list))

            result = []
            for i, term_index in enumerate(combined_list):
                start = term_index + 4
                sort_index = arg_index_list.index(i)
                if sort_index < 4:
                    next_ = arg_index_list[sort_index + 1]
                    re = s[start : combined_list[next_]]
                else:
                    re = s[start:]
                result.append(re.strip())

            ac, sp, at, ot, ie = result

            at = clean_punctuation(at)
            ot = clean_punctuation(ot)

            quints.append((at, ac, sp, ot, ie))

        except KeyError:
            ac, at, sp, ot, ie = "", "", "", "", ""
        except ValueError:
            try:
                print(f"Cannot decode: {s}")
                pass
            except UnicodeEncodeError:
                print(f"A string cannot be decoded")
                pass
            ac, at, sp, ot, ie = "", "", "", "", ""

    return quints


def format_output(output, response_key, response_head):
    output = flatten_output(output)
    formatted_output = []
    for out in output:
        prediction = clean_output(out, response_key, response_head)
        if "[END]" in prediction:
            prediction = prediction[: prediction.find("[END]")]
        prediction = prediction.strip()

        quints = extract_spans(prediction)
        formatted_output.append(quints)

        # prediction, valid = extract_list_str(prediction)
        # if not valid:
        #     formatted_output.append([])
        #     continue

        # prediction = format_list_str(prediction, response_head)
        # formatted_tuples = eval_list_str(prediction)
        # formatted_output.append(formatted_tuples)

    return formatted_output


def dump_output(output_file, formatted_output):
    if "/" in output_file:
        Path(output_file[: output_file.rfind("/")]).mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        dump(formatted_output, f)
