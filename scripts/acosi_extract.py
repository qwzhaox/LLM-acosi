import json
from pipeline import get_model_output
from utils import (
    get_file_path,
    format_output,
    get_args,
    get_formatted_output_w_metadata,
    dump_output,
)

category_file_path = get_file_path("shoes-acosi-cate-list.json")

with open(category_file_path, "r") as f:
    shoes_cate_list = json.load(f)


# categories_with_descriptions = [
#     "shoe (the item in mention)"
#     "shoe.performance (Relating to the performance and function of the item)",
#     "shoe.performance.functional_applicability (Functional ability to enable/perform a specified action or activity; work; operate.)",
#     "shoe.performance.sizing (Relates to the sizing)",
#     "shoe.performance.comfort (Producing or affording physical comfort, support, or ease.)",
#     "shoe.performance.durability (Able to exist for a long time without significant deterioration in quality or value.)",
#     "shoe.performance.stability (The quality, state, or degree of being stable)",
#     "shoe.context_of_use (The situation in which something happens; The group of conditions that exist where and when something happens.)",
#     "shoe.context_of_use.usage_frequency (the frequency of use of a product)",
#     "shoe.context_of_use.use_case (For what is the product used for. )",
#     "shoe.context_of_use.place (Where the product is being used; the user's environment.)",
#     "shoe.context_of_use.review_temporality (A statement on the product and the user's time of use with it.)",
#     "shoe.context_of_use.purchase_context (Other factors that led the user to purchasing the shoe.)",
#     "shoe.appearance (The aesthetic appearance of the shoe. The way that someone or something looks.)",
#     "shoe.appearance.color (A quality such as red, blue, green, yellow, etc., that you see when you look at something; visual perception that enables one to differentiate otherwise identical objects.)",
#     "shoe.appearance.material (The quality or state of being material; an product quality relating to, derived from, or consisting of matter.)",
#     "shoe.appearance.shoe_component (a constituent part; one of the parts that form the shoe)",
#     "shoe.appearance.form (A particular form or shape of an object; the Spatial form or contour of the object.)",
#     "shoe.misc (Things that don't fall in the other categories)",
# ]

# shoes_cate_list = categories_with_descriptions


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
        "Review: omg these are the most comfortable sneakers in the world . i can walk 5 , 6 , 7 miles in them . my whole body may be tired but my feet are great !",
        f"{response_head} [A] sneakers [C] performance#comfort [S] positive [O] omg these are the most comfortable sneakers in the world [I] direct [SSEP] [A] NULL [C] performance#use case applicability [S] positive [O] i can walk 5 , 6 , 7 miles [I] indirect [SSEP] [A] NULL [C] performance#comfort [S] positive [O] i can walk 5 , 6 , 7 miles [I] indirect [SSEP] [A] NULL [C] performance#general [S] positive [O] my whole body may be tired but my feet are great [I] indirect [END]\n\n",
    ]

    examples = [example1, example2]

    return prompt, examples, response_head


def main(args):
    prompt, examples, response_head = get_ACOSI_extract_prompt()
    output, response_key, reviews = get_model_output(
        args, prompt, examples, absa_task="acosi-extract"
    )
    formatted_output, raw_predictions = format_output(
        output, response_key, response_head
    )
    formatted_output_w_metadata = get_formatted_output_w_metadata(
        formatted_output, raw_predictions, reviews
    )
    dump_output(args.output_file, formatted_output)
    dump_output(args.output_file + "_METADATA", formatted_output_w_metadata)


if __name__ == "__main__":
    args = get_args()
    main(args)
