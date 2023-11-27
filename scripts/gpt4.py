import json
import os
from pickle import dump
from pipeline import run_pipeline, get_formatted_annotations, alpaca_format_prompt
from utils import get_file_path, format_output, get_args, dump_output
from acos_extract import get_ACOS_extract_prompt
from acos_extend import get_ACOS_extend_prompt, get_ACOS_annotations, get_ACOSI_annotations
from acosi_extract import get_ACOSI_extract_prompt
from openai import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def main(args):
    dataset_file = args.dataset_file
    absa_task = args.absa_task
    output_file = "data/model_output/gpt4/" + absa_task + "/" + dataset_file + "/output.pkl"

    # get prompts for different tasks
    if absa_task == "acos-extract":
        if "rest" in dataset_file:
            prompt, examples, response_head = get_ACOS_extract_prompt("restaurant")
        elif "laptop" in dataset_file:
            prompt, examples, response_head = get_ACOS_extract_prompt("laptop")
        else:
            raise ValueError("Invalid dataset domain.")
    elif absa_task == "acos-extend":
        prompt, examples, response_head = get_ACOS_extend_prompt()
    elif absa_task == "acosi-extract":
        prompt, examples, response_head = get_ACOSI_extract_prompt()
    else:
        raise ValueError("invalid absa task name.")

    # get lines in dataset file
    lines = []
    with open(dataset_file, 'r') as f:
        for line in f:
            lines.append(line.strip())
    
    # get prompts in each line
    prompts = []
    for line in lines:
        review = line.split("####")[0]
        annotations = eval(line.split("####")[1])

        review_str = f"Review: {review.strip()}\n"
        annotations_str = ""
        if absa_task == "acos-extend":
            annotations_str = (
                f"ACOS quadruples: {get_formatted_annotations(annotations)}\n"
            )

        examples_str = "".join(examples)
        bare_prompt = (
            prompt + examples_str + f"Your Task:\n" + review_str + annotations_str
        )
        formatted_prompt, response_key = alpaca_format_prompt()
        final_prompt = formatted_prompt.format(instruction=bare_prompt)
        prompts.append(final_prompt)


    # run gpt4 for each prompt (can do batches if exceed max rate)
    output = ""
    for prompt in prompts:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
            {"role": "user", "content": prompt}
            ]
        )
        output += completion['choices'][0]['message']['content']


    # output result
    response_key = "### Response:"
    formatted_output = format_output(output, response_key, response_head)
    # format for different tasks
    if absa_task == "acos-extract":
        formatted_output = [[quint[:-1] for quint in quints] for quints in formatted_output]
    elif absa_task == "acos-extend":
        acos_annotations = get_ACOS_annotations(len(formatted_output))
        formatted_output = get_ACOSI_annotations(acos_annotations, formatted_output)

    dump_output(output_file, formatted_output)
    

if __name__ == "__main__":
    args = get_args()
    #absa_task, dataset_file
    main(args)



