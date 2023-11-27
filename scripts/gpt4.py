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

laptop_category_file_path = get_file_path("laptop-acos-cate-list.json")
restaurant_category_file_path = get_file_path("restaurant-acos-cate-list.json")

with open(laptop_category_file_path, "r") as f:
    laptop_cate_list = json.load(f)

with open(restaurant_category_file_path, "r") as f:
    restaurant_cate_list = json.load(f)


def get_ACOS_extract_prompt(dataset_domain):
    prompt = """Given a online customer review, extract the corresponding ACOS (Aspect-Category-Opinion-Sentiment) quadruples.

Each quadruple is comprised of 4 components:
- Aspect [A]: The span of text in the review that indicates the particular aspect that the customer is referring to. Aspects are not always explicitly stated; if this is the case, use a NULL label for the aspect.
- Category [C]: The category of the aspect, selected from the following list: {category_list}
- Sentiment [S]: The polarity of the sentiment: positive, negative, or neutral.
- Opinion [O]: The span of text in the review that indicates the opinion that expresses the sentiment. Opinions are not always explicitly stated; if this is the case, use a NULL lable for the opinion.\n\n
"""

    if dataset_domain == "laptop":
        category_list = "[" + ",".join(laptop_cate_list) + "]"
    elif dataset_domain == "restaurant":
        category_list = "[" + ",".join(restaurant_cate_list) + "]"
    else:
        raise ValueError("Invalid dataset domain.")

    prompt = prompt.format(category_list=category_list)

    response_head = "ACOS quadruples:"

    example1 = f"""Example 1:

Review: the food was lousy - too sweet or too salty and the portions tiny .

Response:
{response_head} [A] food [C] food#quality [S] negative [O] lousy [SSEP] [A] food [C] food#quality [S] negative [O] too sweet [SSEP] [A] food [C] food#quality [S] negative [O] too salty [SSEP] [A] portions [C] food#style_options [S] negative [O] tiny [END]\n\n
"""

    example2 = f"""Example 2:

Review: the decor is night tho . . . but they really need to clean that vent in the ceiling . . . its quite un - appetizing , and kills your effort to make this place look sleek and modern .

Response:
{response_head} [A] place [C] ambience#general [S] negative [O] sleek [SSEP] [A] place [C] ambience#general [S] negative [O] modern [SSEP] [A] decor [C] ambience#general [S] positive [O] night [SSEP] [A] vent [C] ambience#general [S] negative [O] un - appetizing [END]\n\n
"""

    example3 = f"""Example 1:

Review: first one that they shipped was obviously defective , super slow and speakers were garbled .

Response:
{response_head} [A] NULL [C] shipping#general [S] negative [O] defective [SSEP] [A] NULL [C] shipping#general [S] negative [O] slow [SSEP] [A] speakers [C] multimedia_devices#general [S] negative [O] garbled [END]\n\n
"""

    example4 = f"""Example 2:

Review: powers up immediately , great battery life , great keyboard , amazing features .

Response:
{response_head} [A] powers up [C] laptop#operation_performance [S] positive [O] NULL [SSEP] [A] battery life [C] battery#general [S] positive [O] great [SSEP] [A] keyboard [C] keyboard#general [S] positive [O] NULL [SSEP] [A] NULL [C] laptop#design_features [S] positive [O] amazing [END]\n\n
"""

    if dataset_domain == "laptop":
        return prompt, [example1, example2], response_head
    elif dataset_domain == "restaurant":
        return prompt, [example3, example4], response_head
    else:
        raise ValueError("Invalid dataset domain.")


def main(args):
    dataset_file = args.dataset_file
    absa_task = args.absa_task

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

    lines = []
    with open(dataset_file, 'r') as f:
        for line in f:
            lines.append(line.strip())
    
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


    output = ""
    for prompt in prompts:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
            {"role": "user", "content": prompt}
            ]
        )
        output += completion['choices'][0]['message']['content']


    response_key = "### Response:"
    formatted_output = format_output(output, response_key, response_head)
    if absa_task == "acos-extract":
        formatted_output = [[quint[:-1] for quint in quints] for quints in formatted_output]
    elif absa_task == "acos-extend":
        acos_annotations = get_ACOS_annotations(len(formatted_output))
        formatted_output = get_ACOSI_annotations(acos_annotations, formatted_output)
    dump_output(args.output_file, formatted_output)

    formatted_output = format_output(output, response_key, response_head)
    

if __name__ == "__main__":
    args = get_args()
    #absa_task, dataset_file, output_file
    main(args)



