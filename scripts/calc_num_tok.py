from argparse import ArgumentParser
from tqdm import tqdm
from nltk import word_tokenize
from pipeline import get_formatted_annotations
from acos_extend import get_ACOS_extend_prompt
from acos_extract import get_ACOS_extract_prompt
from acosi_extract import get_ACOSI_extract_prompt
from pipeline import get_prompts

parser = ArgumentParser()
parser.add_argument(
    "-d", "--dataset_file", type=str, required=True, help="Dataset file"
)
parser.add_argument("-a", "--absa_task", type=str, required=True, help="Task to run")

args = parser.parse_args()

with open(args.dataset_file, "r") as f:
    dataset = f.readlines()

if args.absa_task == "acos-extend":
    bare_prompt, examples, _ = get_ACOS_extend_prompt()
elif args.absa_task == "acos-extract":
    if "laptop" in args.dataset_file:
        bare_prompt, examples, _ = get_ACOS_extract_prompt("laptop")
    elif "rest" in args.dataset_file:
        bare_prompt, examples, _ = get_ACOS_extract_prompt("restaurant")
    else:
        raise ValueError(
            f"Invalid dataset file {args.dataset_file} for ABSA task {args.absa_task}"
        )
elif args.absa_task == "acosi-extract":
    bare_prompt, examples, _ = get_ACOSI_extract_prompt()
else:
    raise ValueError(f"Invalid ABSA task {args.absa_task}")

prompts, _, _ = get_prompts(args.dataset_file, bare_prompt, examples, args.absa_task)
total_tokens = 0

for prompt in prompts:
    total_tokens += len(word_tokenize(prompt))

print(f"Total input tokens: {total_tokens}")
print(f"Avg tokens per prompt: {total_tokens/len(prompts)}")

if args.absa_task == "acos-extract" or args.absa_task == "acosi-extract":
    total_out_tokens = 0
    for data in dataset:
        annotations = eval(data.split("####")[1])
        annotations_str = f"ACOS quadruples: {get_formatted_annotations(annotations)}\n"
        total_out_tokens += len(word_tokenize(annotations_str))

    print(f"Total output tokens: {total_out_tokens}")
    print(f"Avg out tokens per prompt: {total_out_tokens/len(prompts)}")
