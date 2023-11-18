import json
from acosi_extract_prompt import get_file_path

laptop_category_file_path = get_file_path("laptop-acosi-cate-list.json")
restaurant_category_file_path = get_file_path("restaurant-acosi-cate-list.json")

with open(laptop_category_file_path, "r") as f:
    laptop_cate_list = json.load(f)

with open(restaurant_category_file_path, "r") as f:
    restaurant_cate_list = json.load(f)


def get_ACOS_extend_prompt(dataset_domain):
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

    example2 = """
    Example 2:\n\n

    Review: first one that they shipped was obviously defective , super slow and speakers were garbled .\n

    Response:\n
    ACOS quadruples: [(Aspect: NULL, Category: shipping#general, Sentiment: Negative, Opinion: "defective"), 
                      (Aspect: NULL, Category: shipping#general, Sentiment: Negative, Opinion: "slow"), 
                      (Aspect: "speakers", Category: multimedia_devices#general, Sentiment: Negative, Opinion: "garbled")]
                      \n\n
    """

    examples = [example1, example2]

    return prompt, examples
