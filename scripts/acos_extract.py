import json
from pickle import dump
from pipeline import run_pipeline
from utils import get_file_path, format_output, get_args

laptop_category_file_path = get_file_path("laptop-acosi-cate-list.json")
restaurant_category_file_path = get_file_path("restaurant-acosi-cate-list.json")

with open(laptop_category_file_path, "r") as f:
    laptop_cate_list = json.load(f)

with open(restaurant_category_file_path, "r") as f:
    restaurant_cate_list = json.load(f)


def get_ACOS_extract_prompt(dataset_domain):
    prompt = """
    Given a online customer review, 
    extract the corresponding ACOS (Aspect-Category-Opinion-Sentiment) quadruples. \n
    
    Each quadruple is comprised of 4 components:\n
    - Aspect: The span of text in the review that indicates the particular aspect that the customer is referring to. 
              Aspects are not always explicitly stated; if this is the case, use a NULL label for the aspect.\n
    - Category: The category of the aspect, selected from the following list: {category_list}\n
    - Sentiment: The polarity of the sentiment: Positive, Negative, or Neutral.\n
    - Opinion: The span of text in the review that indicates the opinion that expresses the sentiment.
               Opinions are not always explicitly stated; if this is the case, use a NULL lable for the opinion.\n\n
    """

    if dataset_domain == "laptop":
        category_list = "[" + ",".join(laptop_cate_list) + "]"
    elif dataset_domain == "restaurant":
        category_list = "[" + ",".join(restaurant_cate_list) + "]"
    else:
        raise ValueError("Invalid dataset domain.")

    prompt = prompt.format(category_list=category_list)

    example1 = """
    Example 1:\n\n

    Review: the food was lousy - too sweet or too salty and the portions tiny .\n

    Response:\n
    ACOS quadruples: [(Aspect: "food", Category: food#quality, Sentiment: Negative, Opinion: "lousy"),
                      (Aspect: "food", Category: food#quality, Sentiment: Negative, Opinion: "too sweet"),
                      (Aspect: "food", Category: food#quality, Sentiment: Negative, Opinion: "too salty"),
                      (Aspect: "portions", Category: food#style_options, Sentiment: Negative, Opinion: "tiny")]
                      \n\n
    """

    example2= """
    Example 2:\n\n

    Review: the decor is night tho . . . but they really need to clean that vent in the ceiling . . . its quite un - appetizing , and kills your effort to make this place look sleek and modern .\n
    
    Response:\n
    ACOS quadruples: [(Aspect: "place", Category: ambience#general, Sentiment: Negative, Opinion: "sleek"),
                      (Aspect: "place", Category: ambience#general, Sentiment: Negative, Opinion: "modern"),
                      (Aspect: "decor", Category: ambience#general, Sentiment: Positive, Opinion: "night"),
                      (Aspect: "vent", Category: ambience#general, Sentiment: Negative, Opinion: "un - appetizing")]
                      \n\n
    """

    example3 = """
    Example 2:\n\n

    Review: first one that they shipped was obviously defective , super slow and speakers were garbled .\n

    Response:\n
    ACOS quadruples: [(Aspect: NULL, Category: shipping#general, Sentiment: Negative, Opinion: "defective"), 
                      (Aspect: NULL, Category: shipping#general, Sentiment: Negative, Opinion: "slow"), 
                      (Aspect: "speakers", Category: multimedia_devices#general, Sentiment: Negative, Opinion: "garbled")]
                      \n\n
    """

    example4 = """
    Example 4:\n\n

    Review: powers up immediately , great battery life , great keyboard , amazing features .\n

    Response:\n
    ACOS quadruples: [(Aspect: "powers up", Category: laptop#operation_performance, Sentiment: Positive, Opinion: NULL), 
                      (Aspect: "battery life", Category: battery#general, Sentiment: Positive, Opinion: "great"), 
                      (Aspect: "keyboard", Category: keyboard#general, Sentiment: Positive, Opinion: NULL),
                      (Aspect: NULL, Category: laptop#design_features, Sentiment: Positive, Opinion: "amazing")]
                      \n\n
    """

    if dataset_domain == "laptop":
        return prompt, [example1, example2]
    elif dataset_domain == "restaurant":
        return prompt, [example3, example4]
    else:
        return -1

def main(args):
    prompt, examples_laptop = get_ACOS_extract_prompt("laptop")
    prompt, examples_restaurant = get_ACOS_extract_prompt("restaurant")

    output_laptop, response_key_laptop = run_pipeline(args, prompt, examples_laptop, absa_task="acos_extract")
    output_restaurant, response_key_restaurant = run_pipeline(args, prompt, examples_restaurant, absa_task="acos_extract")
    
    output = output_laptop + output_restaurant
    response_key = response_key_laptop + response_key_restaurant
    
    formatted_output = format_output(output, response_key)
    with open(args.output_file, "w") as f:
        dump(formatted_output, f)


if __name__ == "__main__":
    args = get_args()
    main(args)
