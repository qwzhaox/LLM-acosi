import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
from nltk import word_tokenize
from transformers.generation import GenerationConfig
from pathlib import Path

from utils import (
    get_dataset_domain,
    xu_etal_format_prompt,
    old_format_prompt,
    get_review_str,
    get_examples,
    flatten_output,
    XU_ETAL_INPUT_KEY,
    XU_ETAL_OUTPUT_KEY,
    OLD_REVIEW_KEY,
    OLD_RESPONSE_KEY,
    EXAMPLE_REVIEW,
    EXAMPLE_RESPONSE,
)
from gpt_pipeline import query_gpt
from prompts import PROMPTS, PROMPTS_OLD, CATE_DICT


device = 0 if torch.cuda.is_available() else -1


def get_prompts(
    dataset_file, k_examples=5, limit=None, is_old_prompt=False, absa_task="acosi-extract", model="llama-2"
):
    with open(dataset_file, "r") as f:
        dataset = f.readlines()

    dataset_path = Path(dataset_file).parent
    dataset_domain = get_dataset_domain(dataset_path)

    PROMPT_DICT = PROMPTS_OLD if is_old_prompt else PROMPTS

    instruction = PROMPT_DICT[absa_task]["instruction"]
    context = PROMPT_DICT[absa_task]["context"].format(category_list=CATE_DICT[dataset_domain])
    output_format = PROMPT_DICT[absa_task]["output-format"]

    if is_old_prompt:
        formatted_prompt = old_format_prompt(instruction, context, output_format)
        review_key = OLD_REVIEW_KEY
        response_key = OLD_RESPONSE_KEY
    else:
        formatted_prompt = xu_etal_format_prompt(instruction, context, output_format)
        review_key = XU_ETAL_INPUT_KEY
        response_key = XU_ETAL_OUTPUT_KEY

    print("Processing dataset...")

    prompts = []
    if "gpt" in model.lower():
        prompt_instr = {
                "instruction": instruction,
                "context": context,
                "output-format": output_format,
                }
        prompts.append(prompt_instr)
    reviews = []

    for data in tqdm(dataset, desc="Processing Dataset", unit="item"):
        review = data.split("####")[0]
        reviews.append(review)

        annotations = eval(data.split("####")[1])

        review_str = get_review_str(review, absa_task, annotations, is_old_prompt)
        examples = get_examples(dataset_path, absa_task, is_old_prompt=is_old_prompt, k_examples=k_examples, limit=limit)

        if "gpt" in model.lower():
            prompt = {
                "examples": examples,
                "review": review_str,
            }
        else:
            example_str = ""
            for example in examples:
                if is_old_prompt:
                    example_str += formatted_prompt
                example_str += f"{review_key} {example[EXAMPLE_REVIEW]}\n"
                example_str += f"{response_key} {example[EXAMPLE_RESPONSE]}\n"
            if is_old_prompt:
                prompt = f"{example_str}{formatted_prompt}{OLD_REVIEW_KEY}{review_str}{OLD_RESPONSE_KEY}"
            else:
                prompt = f"{formatted_prompt}{example_str}{XU_ETAL_INPUT_KEY}{review_str}{XU_ETAL_OUTPUT_KEY}"

        prompts.append(prompt)

    return prompts, reviews


def run_pipeline(args, prompts):
    # Initialize the pipeline with the specified model, and set the device
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, model_max_length=args.max_length
    )
    # model = AutoModelForCausalLM.from_pretrained(args.model_name)
    gen_config = GenerationConfig.from_pretrained(args.model_name)
    gen_config.max_new_tokens = args.max_new_tokens
    gen_config.max_length = args.max_length
    # pre_config = PretrainedConfig.from_pretrained(args.model_name)

    print("Initializing pipeline...")

    model_pipe = pipeline(
        args.task,
        model=args.model_name,
        tokenizer=tokenizer,
        device_map="auto",
        trust_remote_code=args.remote,
        # config=pre_config,
    )

    print("Running pipeline...")

    output = model_pipe(prompts, generation_config=gen_config)

    flat_output = flatten_output(output)
    total_out_tokens = 0
    for out in flat_output:
        total_out_tokens += len(word_tokenize(out["generated_text"].strip()))
    print(f"Total output tokens: {total_out_tokens}")
    print(f"Avg out tokens per prompt: {total_out_tokens/len(prompts)}")

    return output


def get_model_output(args):
    prompts, reviews = get_prompts(
        args.dataset_file,
        k_examples=args.k_examples,
        limit=args.limit,
        is_old_prompt=args.is_old_prompt,
        absa_task=args.absa_task,
        model=args.model_name.lower(),
    )
    if "gpt" in args.model_name.lower():
        output = query_gpt(
            args.model_name,
            prompts,
            max_tokens=args.max_new_tokens,
            is_old_prompt=args.is_old_prompt,
        )
    else:
        output = run_pipeline(args, prompts)

    return output, reviews
