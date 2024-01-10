import re
import pickle
import json
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from itertools import chain
from string import punctuation

ALPACA_INSTRUCTION_KEY = "### Instruction:"
ALPACA_RESPONSE_KEY = "### Response:"
ALPACA_INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

EXAMPLE_REVIEW = 0
EXAMPLE_RESPONSE = 1


SSEP = "SSEP"
END = "END"
ACOSI = "ACOSI"


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
        "--tokenizer_name", type=str, default=None, help="Tokenizer name"
    )
    parser.add_argument("--task", type=str, default=None, help="Task name")
    parser.add_argument(
        "--remote", action="store_true", help="Whether to trust remote code"
    )
    parser.add_argument("--max_length", type=int, default=1024, help="Max length")
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Max new tokens"
    )
    parser.add_argument("--dataset_file", type=str, required=True, help="Dataset file")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    args = parser.parse_args()
    return args


def alpaca_format_prompt():
    formatted_prompt = """
{instruction_key}
{instruction}
{response}
""".format(
        instruction_key=ALPACA_INSTRUCTION_KEY,
        instruction="{instruction}",
        response=ALPACA_RESPONSE_KEY,
    )
    return formatted_prompt, ALPACA_RESPONSE_KEY, ALPACA_INTRO_BLURB


def alpaca_format_prompt_w_header():
    prompt_sans_header, _, intro_blurb = alpaca_format_prompt()
    prompt_for_generation_format = intro_blurb + "\n" + prompt_sans_header
    return prompt_for_generation_format, ALPACA_RESPONSE_KEY


def flatten_output(output):
    if type(output[0]) is list:
        return list(chain(*output))
    return output


def needs_closing_bracket(idx, prediction):
    return idx == len(prediction) or idx < len(prediction) and prediction[idx] != "]"


def is_special_token(start_idx, end_idx, prediction, special_tokens):
    return (
        end_idx <= len(prediction) and prediction[start_idx:end_idx] in special_tokens
    )


def fix_brackets(prediction):
    insert_loc = []
    delete_loc = len(prediction)

    for i, char in enumerate(prediction):
        if char == "[":
            start = i + 1
            ssep_end = start + len(SSEP)
            end_end = start + len(END)
            acosi_end = start + 1
            if is_special_token(start, ssep_end, prediction, SSEP):
                if needs_closing_bracket(ssep_end, prediction):
                    insert_loc.append(ssep_end)
            elif is_special_token(start, end_end, prediction, END):
                if needs_closing_bracket(end_end, prediction):
                    insert_loc.append(end_end)
            elif is_special_token(start, acosi_end, prediction, ACOSI):
                if needs_closing_bracket(acosi_end, prediction):
                    insert_loc.append(acosi_end)
            elif i + 1 == len(prediction):
                delete_loc = i

    prediction = prediction[:delete_loc]
    for i in reversed(insert_loc):
        prediction = prediction[:i] + "]" + prediction[i:]

    return prediction


def clean_output(out, response_key, response_head):
    try:
        prediction = out["generated_text"].strip()
    except:
        prediction = out.strip()

    if response_key in prediction:
        prediction = prediction[prediction.rfind(response_key) :].strip()
        prediction = prediction[len(response_key) :].strip()
    if response_head in prediction:
        prediction = prediction[prediction.rfind(response_head) :].strip()
        prediction = prediction[len(response_head) :].strip()
    if "\n" in prediction:
        prediction = prediction[: prediction.find("\n")]

    print("Raw prediction: ", prediction)

    raw_prediction = prediction
    prediction = fix_brackets(prediction)

    if "[END]" in prediction:
        prediction = prediction[: prediction.find("[END]")]
    prediction = prediction.strip()

    return prediction, raw_prediction


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

            is_all_null = True

            for term in [ac, sp, at, ot, ie]:
                if term.lower() != "null" and term != "":
                    is_all_null = False
                    break
            if not is_all_null:
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
    raw_predictions = []
    for out in output:
        prediction, raw_prediction = clean_output(out, response_key, response_head)

        print("Prediction: ", prediction)

        quints = extract_spans(prediction)
        formatted_output.append(quints)
        raw_predictions.append(raw_prediction)

        print("Quints: ", quints)

    return formatted_output, raw_predictions


def get_formatted_output_w_metadata(formatted_output, raw_predictions, reviews):
    formatted_output_w_metadata = []
    if len(formatted_output) == len(raw_predictions) == len(reviews):
        for annotation, raw_prediction, review in zip(
            formatted_output, raw_predictions, reviews
        ):
            annotation_w_metadatum = {}
            annotation_w_metadatum["annotation"] = annotation
            annotation_w_metadatum["review"] = review
            annotation_w_metadatum["raw_predictions"] = raw_prediction
            formatted_output_w_metadata.append(annotation_w_metadatum)
    else:
        print(
            f"Length of formatted output {len(formatted_output)} does not match corresponding lengths: {len(raw_predictions)}, {len(reviews)}"
        )
    return formatted_output_w_metadata


def dump_output(output_file, formatted_output):
    output_file = (
        output_file + ".pkl"
        if "_METADATA" not in output_file
        else output_file + ".json"
    )
    if "/" in output_file:
        Path(output_file[: output_file.rfind("/")]).mkdir(parents=True, exist_ok=True)

    if ".pkl" in output_file:
        with open(output_file, "wb") as f:
            pickle.dump(formatted_output, f)
    elif ".json" in output_file:
        with open(output_file, "w") as f:
            json.dump(formatted_output, f, indent=4)
