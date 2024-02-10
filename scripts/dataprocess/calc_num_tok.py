from argparse import ArgumentParser
from nltk import word_tokenize
from sys import path
path.insert(1, './scripts/run_llm/')
from pipeline import get_prompts
from utils import get_xu_etal_formatted_annotations, get_old_formatted_annotations

parser = ArgumentParser()
parser.add_argument(
    "-d", "--dataset_file", type=str, required=True, help="Dataset file"
)
parser.add_argument("-a", "--absa_task", type=str, required=True, help="Task to run")
parser.add_argument("-o", "--is_old_prompt", action="store_true", help="Use old prompt")

args = parser.parse_args()

with open(args.dataset_file, "r") as f:
    dataset = f.readlines()
get_formatted_annotations = get_old_formatted_annotations if args.is_old_prompt else get_xu_etal_formatted_annotations
prompts, _ = get_prompts(args.dataset_file, k_examples=5, limit=None, is_old_prompt=args.is_old_prompt, absa_task=args.absa_task, model="llama-2")
total_tokens = 0

for prompt in prompts:
    print(prompt)
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
