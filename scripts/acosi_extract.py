import json
from pipeline import run_pipeline
from pickle import dump
from utils import get_file_path, format_output, get_args, dump_output

category_file_path = get_file_path("shoes-acosi-cate-list.json")

with open(category_file_path, "r") as f:
    shoes_cate_list = json.load(f)


def get_ACOSI_extract_prompt():
    prompt = """Given a product review, extract the corresponding ACOSI (Aspect-Category-Opinion-Sentiment-Implicit/Explicit) quintuples.

Each quintuple is comprised of 5 components:
- Aspect [A]: The span of text in the review that indicates the particular aspect that the customer is referring to. Aspects are not always explicitly stated; if this is the case, use a NULL label for the aspect.
- Category [C]: The category of the aspect, selected from the following list: {category_list}
- Sentiment [S]: The polarity of the sentiment: positive, negative, or neutral.
- Opinion [O]: The span of text in the review that indicates the opinion that expresses the sentiment. Opinions are not always explicitly stated; if this is the case, please try to identify the span of text that best expresses the sentiment implicitly.
- Implicit Indicator [I]: Indicates whether the opinion is implicit or explicit (indirect or direct).

[END] is used to mark the end of the set of quintuples associated with a review and [SSEP] is used to separate individual quintuples in the set.\n
"""

    category_list = "[" + ",".join(shoes_cate_list) + "]"

    prompt = prompt.format(category_list=category_list)

    response_head = "ACOSI quintuples:"

    example1 = [
        "Review: the design is great poor color choices too bland . color choices from previous shoes was much better .\n",
        f"{response_head} [A] NULL [C] appearance#form [S] positive [O] design is great [I] direct [SSEP] [A] NULL [C] appearance#color [S] negative [O] poor color choices [I] direct [SSEP] [A] shoes [C] appearance#color [S] negative [O] color choices from previous shoes was much better [I] indirect [END]\n\n",
    ]

    example2 = [
        "Review: had to order a larger size than what i normally wear . shoe would be better if offered as an adjustable shoe . shoe is overpriced for quality . i bought cheaper slides in the past that were more comfortable .",
        f"{response_head} [A] NULL [C] performance#sizing_fit [S] neutral [O] had to order a larger size than what i normally wear [I] direct [SSEP] [A] NULL [C] contextofuse#purchase\\\\_context [S] negative [O] had to order a larger size than what i normally wear [I] direct [SSEP] [A] shoe [C] appearance#form [S] neutral [O] would be better if offered as an adjustable shoe [I] direct [SSEP] [A] shoe [C] cost/value [S] negative [O] overpriced for quality [I] direct [SSEP] [A] slides [C] cost/value [S] negative [O] i bought cheaper slides in the past that were more comfortable [I] direct [SSEP] [A] slides [C] performance#comfort [S] negative [O] i bought cheaper slides in the past that were more comfortable [I] direct [END]\n\n",
    ]

    examples = [example1, example2]

    return prompt, examples, response_head


def main(args):
    prompt, examples, response_head = get_ACOSI_extract_prompt()

    if "gpt" in args.model_name.lower():
        pass
    else:
        output, response_key = run_pipeline(
            args, prompt, examples, absa_task="acosi-extract"
        )
    formatted_output = format_output(output, response_key, response_head)
    dump_output(args.output_file, formatted_output)


if __name__ == "__main__":
    args = get_args()
    main(args)
