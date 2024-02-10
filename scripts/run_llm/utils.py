import re
import pickle
import json
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from itertools import chain
from string import punctuation
from random import sample
from prompts import PROMPTS_OLD

OLD_INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
OLD_INSTRUCTION_KEY = "### Instruction:"
OLD_CONTEXT_KEY = "Context:"
OLD_OUTPUT_FORMAT_KEY = "Output format:"
OLD_REVIEW_KEY = "Review:"
OLD_RESPONSE_KEY = "### Response:"

XU_ETAL_INSTRUCTION_KEY = "Instruction:"
XU_ETAL_CONTEXT_KEY = "Context:"
XU_ETAL_OUTPUT_FORMAT_KEY = "Output format:"
XU_ETAL_INPUT_KEY = "Input:"
XU_ETAL_OUTPUT_KEY = "Output:"

EXAMPLE_REVIEW = 0
EXAMPLE_RESPONSE = 1

ASPECT_IDX = 0
CATEGORY_IDX = 1
SENTIMENT_IDX = 2
OPINION_IDX = 3
IMPLICIT_INDICATOR_IDX = 4

SSEP = "SSEP"
END = "END"
ACOSI = "ACOSI"


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="LLM model name")
    parser.add_argument(
        "--tokenizer_name", type=str, default=None, help="Tokenizer name"
    )
    parser.add_argument("--task", type=str, default=None, help="Task name")
    parser.add_argument("--absa_task", type=str, default=None, help="ABSA task name")
    parser.add_argument(
        "--remote", action="store_true", help="Whether to trust remote code"
    )
    parser.add_argument("--max_length", type=int, default=1024, help="Max length")
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Max new tokens"
    )
    parser.add_argument("--dataset_file", type=str, required=True, help="Dataset file")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument("--k_examples", type=int, default=1, help="Number of examples")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples to choose from.")
    parser.add_argument("--is_old_prompt", action="store_true", help="Use old prompt format")
    args = parser.parse_args()
    return args


def get_dataset_domain(dataset_path):
    if "rest" in str(dataset_path):
        dataset_domain = "rest"
    elif "laptop" in str(dataset_path):
        dataset_domain = "laptop"
    elif "shoes" in str(dataset_path):
        dataset_domain = "shoes"
    else:
        raise ValueError("Invalid dataset domain.")
    return dataset_domain


def xu_etal_format_prompt(instruction, context, output_format):
    formatted_prompt = """{instruction}
{context}
{output_format}
""".format(
        instruction=f"{XU_ETAL_INSTRUCTION_KEY} {instruction}",
        context=f"{XU_ETAL_CONTEXT_KEY} {context}",
        output_format=f"{XU_ETAL_OUTPUT_FORMAT_KEY} {output_format}",
        )
    return formatted_prompt


def get_xu_etal_formatted_annotations(annotations, opinion_span_only=False):
    annots = []
    for annotation in annotations:
        category = annotation[CATEGORY_IDX].replace('#', ' ').replace('\\_', '_').lower()
        if opinion_span_only:
            new_annot_str = f"({annotation[OPINION_IDX]})"
        elif len(annotation) == 4:
            # new_annot_str = f"(Aspect: {annotation[ASPECT_IDX]}, Category: {annotation[CATEGORY_IDX]}, Sentiment: {annotation[SENTIMENT_IDX]}, Opinion: {annotation[OPINION_IDX]})"
            new_annot_str = f"({annotation[ASPECT_IDX]}, {category}, {annotation[SENTIMENT_IDX]}, {annotation[OPINION_IDX]})"
        elif len(annotation) == 5:
            new_annot_str = f"({annotation[ASPECT_IDX]}, {category}, {annotation[SENTIMENT_IDX]}, {annotation[OPINION_IDX]}, {annotation[IMPLICIT_INDICATOR_IDX]})"

        annots.append(new_annot_str)

    annots_str = "[" + ",".join(annots) + "]"
    return annots_str


def old_format_prompt(instruction, context, output_format):
    formatted_prompt = """{intro_blurb}
{instruction_key}
{instruction}
{context}
{output_format}
""".format(
        intro_blurb=OLD_INTRO_BLURB,
        instruction_key=OLD_INSTRUCTION_KEY,
        instruction=instruction,
        context=f"{OLD_CONTEXT_KEY} {context}",
        output_format=f"{OLD_OUTPUT_FORMAT_KEY} {output_format}",
        )
    return formatted_prompt


def get_old_formatted_annotations(annotations):
    annots = []
    for annotation in annotations:
        if len(annotation) == 4:
            # new_annot_str = f"(Aspect: {annotation[ASPECT_IDX]}, Category: {annotation[CATEGORY_IDX]}, Sentiment: {annotation[SENTIMENT_IDX]}, Opinion: {annotation[OPINION_IDX]})"
            new_annot_str = f"[A] {annotation[ASPECT_IDX]} [C] {annotation[CATEGORY_IDX]} [S] {annotation[SENTIMENT_IDX]} [O] {annotation[OPINION_IDX]}"
        elif len(annotation) == 5:
            new_annot_str = f"[A] {annotation[ASPECT_IDX]} [C] {annotation[CATEGORY_IDX]} [S] {annotation[SENTIMENT_IDX]} [O] {annotation[OPINION_IDX]} [I] {annotation[IMPLICIT_INDICATOR_IDX]}"

        annots.append(new_annot_str)

    annots_str = " [SSEP] ".join(annots) + " [END]"

    return annots_str

def get_review_str(review, absa_task, annotations, old_prompt=False):
    review_str = f"{review.strip()}\n"
    annotations_str = ""
    if absa_task == "acos-extend":
        if old_prompt:
            annotations_str = f"ACOS quadruples: {get_old_formatted_annotations(annotations)}\n"
        else:
            annotations_str = f"ACOS quadruples: {get_xu_etal_formatted_annotations(annotations)}\n"

    review_str = review_str + annotations_str
    return review_str


def get_examples(dataset_path, absa_task, is_old_prompt=False, k_examples=5, limit=None):
    examples = []

    if is_old_prompt:
        dataset = dataset_path.name
        for example in PROMPTS_OLD[absa_task]["examples"][dataset]:
            if absa_task == "acos-extend":
                review = f"{example['review']}\nACOS quadruples: {example['acos-quadruples']}"
                response = f"{example['response']}\n{example['response-explanation']}"
            else:
                review = example["review"]
                response = example["response"]

            examples.append((review, response))

        return examples

    with open(Path(dataset_path / "train.txt"), "r") as f:
        train_dataset = f.readlines()
    
    response_train_dataset_path = Path(str(dataset_path).replace('acos', 'acosi'))
    response_train_dataset = []
    if absa_task == "acos-extend":
        with open(Path(response_train_dataset_path / "train.txt"), "r") as f:
            response_train_dataset = f.readlines()

    if limit:
        limit_indices = sample(range(len(train_dataset)), k=limit)
        train_dataset = [train_dataset[i] for i in limit_indices]
        response_train_dataset = [response_train_dataset[i] for i in limit_indices]

    indices = sample(range(len(train_dataset)), k=k_examples)

    for idx in indices:
        review = train_dataset[idx].split("####")[0]
        annotations = eval(train_dataset[idx].split("####")[1])
        annotations = get_xu_etal_formatted_annotations(annotations)

        if absa_task == "acos-extend":
            review = f"{review}\nACOS annotations: {annotations}"
            annotations = eval(response_train_dataset[idx].split("####")[1])
            response = get_xu_etal_formatted_annotations(annotations, opinion_span_only=True)
        else:
            response = annotations
        
        examples.append((review, response))
    
    return examples


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


def clean_output(out, response_key):
    try:
        prediction = out["generated_text"].strip()
    except:
        prediction = out.strip()

    if response_key in prediction:
        prediction = prediction[prediction.rfind(response_key)+len(response_key):].strip()
    if ":" in prediction:
        prediction = prediction[prediction.rfind(":") + 1:].strip()

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


def clean_terms(at, ac, sp, ot, ie):
    at = clean_punctuation(at)
    ot = clean_punctuation(ot)

    is_all_null = True

    for term in [at, ac, sp, ot, ie]:
        if term.lower() != "null" and term != "":
            is_all_null = False
            break

    return at, ac, sp, ot, ie, is_all_null


def extract_spans(seq):
    quints = []
    try:
        sents = eval(seq)
    except:
        return quints

    for s in sents:
        if len(s) == 1:
            ot = s[0]
            ac, sp, at, ie = "null", "null", "null", "null"
        if len(s) == 4:
            at, ac, sp, ot = s
            ie = "null"
        if len(s) == 5:
            at, ac, sp, ot, ie = s

        at, ac, sp, ot, ie, is_all_null = clean_terms(at, ac, sp, ot, ie)
        
        if not is_all_null:
            quints.append((at, ac, sp, ot, ie))

    return quints


def extract_spans_old(seq):
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
            at, ac, sp, ot, ie, is_all_null = clean_terms(at, ac, sp, ot, ie)

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


def format_output(output, is_old_prompt=False):
    response_key = OLD_RESPONSE_KEY if is_old_prompt else XU_ETAL_OUTPUT_KEY

    output = flatten_output(output)
    formatted_output = []
    raw_predictions = []
    for out in output:
        prediction, raw_prediction = clean_output(out, response_key)

        print("Prediction: ", prediction)
        quints = extract_spans_old(prediction) if is_old_prompt else extract_spans_old(prediction)

        formatted_output.append(quints)
        raw_predictions.append(raw_prediction)

        print("Quints: ", quints)

    return formatted_output, raw_predictions


def get_formatted_output_and_metadata(formatted_output, raw_predictions, reviews):
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
