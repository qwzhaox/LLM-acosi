from argparse import ArgumentParser
from tqdm import tqdm
from nltk import word_tokenize
from pipeline import alpaca_format_prompt, get_formatted_annotations
from acos_extend import get_ACOS_extend_prompt
from acos_extract import get_ACOS_extract_prompt
from acosi_extract import get_ACOSI_extract_prompt

parser = ArgumentParser()
parser.add_argument(
    "-d", "--dataset_file", type=str, required=True, help="Dataset file"
)
parser.add_argument("-a", "--absa_task", type=str, required=True, help="Task to run")

args = parser.parse_args()

with open(args.dataset_file, "r") as f:
    dataset = f.readlines()

if args.absa_task == "acos-extend":
    prompt, examples, _ = get_ACOS_extend_prompt()
elif args.absa_task == "acos-extract":
    if "laptop" in args.dataset_file:
        prompt, examples, _ = get_ACOS_extract_prompt("laptop")
    elif "rest" in args.dataset_file:
        prompt, examples, _ = get_ACOS_extract_prompt("restaurant")
    else:
        raise ValueError(
            f"Invalid dataset file {args.dataset_file} for ABSA task {args.absa_task}"
        )
elif args.absa_task == "acosi-extract":
    prompt, examples, _ = get_ACOSI_extract_prompt()
else:
    raise ValueError(f"Invalid ABSA task {args.absa_task}")

formatted_prompt, response_key = alpaca_format_prompt()

prompts = []
total_tokens = 0
total_out_tokens = 0

for i, data in enumerate(tqdm(dataset, desc="Processing", unit="item")):
    review = data.split("####")[0]
    annotations = eval(data.split("####")[1])

    review_str = f"Review: {review.strip()}\n"
    annotations_str = ""
    output_str = ""

    if args.absa_task == "acos-extend":
        annotations_str = f"ACOS quadruples: {get_formatted_annotations(annotations)}\n"
    elif args.absa_task == "acos-extract":
        output_str = f"ACOS quadruples: {get_formatted_annotations(annotations)}\n"
        total_out_tokens += len(word_tokenize(output_str))
    elif args.absa_task == "acosi-extract":
        output_str = f"ACOSI quintuples: {get_formatted_annotations(annotations)}\n"
        total_out_tokens += len(word_tokenize(output_str))

    examples_str = "".join(examples)
    bare_prompt = (
        prompt + examples_str + f"Your Task {i}:\n" + review_str + annotations_str
    )
    final_prompt = formatted_prompt.format(instruction=bare_prompt)
    prompts.append(final_prompt)
    total_tokens += len(word_tokenize(final_prompt))

print(f"Total input tokens: {total_tokens}")
print(f"Avg tokens per prompt: {total_tokens/len(prompts)}")

if args.absa_task == "acos-extract" or args.absa_task == "acosi-extract":
    print(f"Total output tokens: {total_out_tokens}")
    print(f"Avg out tokens per prompt: {total_out_tokens/len(prompts)}")
