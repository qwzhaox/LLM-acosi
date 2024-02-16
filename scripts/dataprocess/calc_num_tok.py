from argparse import ArgumentParser
from nltk import word_tokenize
from sys import path
path.insert(1, './scripts/run_llm/')
from pipeline import Pipeline
from utils import get_xu_etal_formatted_annotations, get_old_formatted_annotations
parser = ArgumentParser()
parser.add_argument(
    "-d", "--dataset_file", type=str, default="data/acosi_dataset/shoes/test.txt", help="Dataset file"
)
parser.add_argument("-a", "--absa_task", type=str, default="acosi-extract", help="Task to run")
parser.add_argument("-o", "--is_old_prompt", action="store_true", help="Use old prompt")
parser.add_argument("-c", "--is_combo_prompt", action="store_true", help="Use old prompt")

args = parser.parse_args()

class Args:
    def __init__(self, dataset_file, absa_task, is_old_prompt, is_combo_prompt):
        self.dataset_file = dataset_file
        self.absa_task = absa_task
        self.is_old_prompt = is_old_prompt
        self.is_combo_prompt = is_combo_prompt

        self.k_examples = 10
        self.limit = None
        self.selection_method = "tf-idf"

        self.model_name = "llama-2"
        self.task = "text-generation"
        self.tokenizer_name = "llama-2"
        self.max_length = 512
        self.max_new_tokens = 100
        self.remote = False

if args.is_old_prompt:
    get_formatted_annotations = get_old_formatted_annotations
else:
    get_formatted_annotations = get_xu_etal_formatted_annotations

pipeline = Pipeline(Args(args.dataset_file, args.absa_task, args.is_old_prompt, args.is_combo_prompt))
prompts, _ = pipeline.get_prompts()
total_tokens = 0
max_tokens = 0

for prompt in prompts:
    print(prompt)
    total_tokens += len(word_tokenize(prompt))
    max_tokens = max(max_tokens, len(word_tokenize(prompt)))

print(f"Total input tokens: {total_tokens}")
print(f"Avg tokens per prompt: {total_tokens/len(prompts)}")
print(f"Max tokens per prompt: {max_tokens}")

with open(args.dataset_file, "r") as f:
    dataset = f.readlines()

if args.absa_task == "acos-extract" or args.absa_task == "acosi-extract" or (args.absa_task == "acos-extend" and "shoes" in args.dataset_file):
    if args.absa_task == "acos-extend":
        dataset_file = args.dataset_file.replace("acos", "acosi")
        with open(dataset_file, "r") as f:
            dataset = f.readlines()
    total_out_tokens = 0
    max_out_tokens = 0
    for data in dataset:
        annotations = eval(data.split("####")[1])
        annotations_str = f"ACOS quadruples: {get_formatted_annotations(annotations)}\n"
        total_out_tokens += len(word_tokenize(annotations_str))
        max_out_tokens = max(max_out_tokens, len(word_tokenize(annotations_str)))

    print(f"Total output tokens: {total_out_tokens}")
    print(f"Avg out tokens per prompt: {total_out_tokens/len(prompts)}")
    print(f"Max out tokens per prompt: {max_out_tokens}")
