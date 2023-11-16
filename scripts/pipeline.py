import argparse
import json
import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

device = 0 if torch.cuda.is_available() else -1


def dolly_15k_format_prompt(prompt):
    instruction_key = "### Instruction:"
    response_key = "### Response:"
    intro_blurb = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    prompt_for_generation_format = """{intro}
                                      {instruction_key}
                                      {instruction}
                                      {response_key}
                                    """.format(
        intro=intro_blurb,
        instruction_key=instruction_key,
        instruction=prompt,
        response_key=response_key,
    )
    return prompt_for_generation_format, response_key


def run_pipeline(args, prompt):
    # Initialize the pipeline with the specified model, and set the device
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    model_pipe = pipeline(
        args.task,
        model=args.model_name,
        tokenizer=args.tokenizer,
        device_map="auto",
        trust_remote_code=args.remote,
    )

    with open(args.dataset_file, "r") as f:
        dataset = f.readlines()

    formatted_prompt, response_key = dolly_15k_format_prompt(prompt)

    output = []

    for data in tqdm(dataset, desc="Processing", unit="item"):
        pass

    return output
