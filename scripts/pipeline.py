import argparse
import json
import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

device = 0 if torch.cuda.is_available() else -1


def dolly_15k_format_prompt():
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
        instruction="{prompt}",
        response_key=response_key,
    )
    return prompt_for_generation_format, response_key


def get_formatted_annotations(annotations):
    for annotation in annotations:
        pass


def run_pipeline(args, prompt, examples=[], absa_task="extract-acosi"):
    # Initialize the pipeline with the specified model, and set the device
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    model_pipe = pipeline(
        args.task,
        model=args.model_name,
        tokenizer=tokenizer,
        device_map="auto",
        trust_remote_code=args.remote,
    )

    with open(args.dataset_file, "r") as f:
        dataset = f.readlines()

    formatted_prompt, response_key = dolly_15k_format_prompt()

    output = []

    for i, data in enumerate(tqdm(dataset, desc="Processing", unit="item")):
        review = data.split("####")[0]
        annotations = data.split("####")[1]

        review_str = f"Review: {review.strip()}\n"
        annotations_str = ""
        if absa_task == "extend":
            annotations_str = (
                f"ACOS quadruples: {get_formatted_annotations(annotations)}\n"
            )

        examples_str = "".join(examples)
        bare_prompt = (
            prompt + examples_str + f"TASK {i}:\n" + review_str + annotations_str
        )
        final_prompt = formatted_prompt.format(bare_prompt)

        output.append(model_pipe(final_prompt, max_new_tokens=args.max_new_tokens))

    return output, response_key
