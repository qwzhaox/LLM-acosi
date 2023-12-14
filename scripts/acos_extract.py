import json
from pickle import dump
from pipeline import get_model_output
from utils import get_file_path, format_output, get_args, dump_output

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
- Opinion [O]: The span of text in the review that indicates the opinion that expresses the sentiment. Opinions are not always explicitly stated; if this is the case, use a NULL label for the opinion.

[END] is used to mark the end of the set of quadruples associated with a review and [SSEP] is used to separate individual quadruples in the set.\n
"""

    if dataset_domain == "laptop":
        category_list = "[" + ",".join(laptop_cate_list) + "]"
    elif dataset_domain == "restaurant":
        category_list = "[" + ",".join(restaurant_cate_list) + "]"
    else:
        raise ValueError("Invalid dataset domain.")

    prompt = prompt.format(category_list=category_list)

    response_head = "ACOS quadruples:"

    example1 = [
        "Review: the food was lousy - too sweet or too salty and the portions tiny .\n",
        f"{response_head} [A] food [C] food#quality [S] negative [O] lousy [SSEP] [A] food [C] food#quality [S] negative [O] too sweet [SSEP] [A] food [C] food#quality [S] negative [O] too salty [SSEP] [A] portions [C] food#style_options [S] negative [O] tiny [END]\n\n",
    ]

    example2 = [
        "Review: the decor is night tho . . . but they really need to clean that vent in the ceiling . . . its quite un - appetizing , and kills your effort to make this place look sleek and modern .\n",
        f"{response_head} [A] place [C] ambience#general [S] negative [O] sleek [SSEP] [A] place [C] ambience#general [S] negative [O] modern [SSEP] [A] decor [C] ambience#general [S] positive [O] night [SSEP] [A] vent [C] ambience#general [S] negative [O] un - appetizing [END]\n\n",
    ]

    example3 = [
        "Review: first one that they shipped was obviously defective , super slow and speakers were garbled .\n",
        f"{response_head} [A] NULL [C] shipping#general [S] negative [O] defective [SSEP] [A] NULL [C] shipping#general [S] negative [O] slow [SSEP] [A] speakers [C] multimedia_devices#general [S] negative [O] garbled [END]\n\n",
    ]

    example4 = [
        "Review: powers up immediately , great battery life , great keyboard , amazing features .\n",
        f"{response_head} [A] powers up [C] laptop#operation_performance [S] positive [O] NULL [SSEP] [A] battery life [C] battery#general [S] positive [O] great [SSEP] [A] keyboard [C] keyboard#general [S] positive [O] NULL [SSEP] [A] NULL [C] laptop#design_features [S] positive [O] amazing [END]\n\n",
    ]

    if dataset_domain == "laptop":
        return prompt, [example1, example2], response_head
    elif dataset_domain == "restaurant":
        return prompt, [example3, example4], response_head
    else:
        raise ValueError("Invalid dataset domain.")


def main(args):
    if "rest" in args.dataset_file:
        prompt, examples, response_head = get_ACOS_extract_prompt("restaurant")
    elif "laptop" in args.dataset_file:
        prompt, examples, response_head = get_ACOS_extract_prompt("laptop")
    else:
        raise ValueError("Invalid dataset domain.")

    output, response_key = get_model_output(
        args, prompt, examples, absa_task="acos-extract"
    )
    formatted_output = format_output(output, response_key, response_head)
    formatted_output = [[quint[:-1] for quint in quints] for quints in formatted_output]

    dump_output(args.output_file, formatted_output)


if __name__ == "__main__":
    args = get_args()
    main(args)
