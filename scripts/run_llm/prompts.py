import json
from pathlib import Path

def get_file_path(file_name):
    # Search in the current directory and all subdirectories
    for path in Path(".").rglob(file_name):
        # Return the first match
        return path
    # Return None if no match is found
    return None

laptop_category_file_path = get_file_path("laptop-acos-cate-list.json")
restaurant_category_file_path = get_file_path("restaurant-acos-cate-list.json")
shoes_category_file_path = get_file_path("shoes-acosi-cate-list-original.json")

with open(laptop_category_file_path, "r") as f:
    laptop_cate_list = json.load(f)

with open(restaurant_category_file_path, "r") as f:
    restaurant_cate_list = json.load(f)

with open(shoes_category_file_path, "r") as f:
    shoes_cate_list = json.load(f)

cate_lists = [laptop_cate_list, restaurant_cate_list, shoes_cate_list]
for i, cate_list in enumerate(cate_lists):
    for j, cate in enumerate(cate_list):
        cate_lists[i][j] = cate.replace("#", " ").replace("\\_", "_").lower()

CATE_DICT = {
    "laptop": laptop_cate_list,
    "rest": restaurant_cate_list,
    "shoes": shoes_cate_list,
}

PROMPTS = {
    "acos-extract": {
        "instruction": "Extract aspect-category-sentiment-opinion quadruples from input data.",
        "context": "An aspect or opinion must be a term existing in input data or null if non-existing; the category is one in the predefined list {category_list}; the sentiment is positive, negative or neutral; do not ask me for more information, I am unable to provide it, and just try your best to finish the task. You can learn from the following examples.",
        "output-format": "(aspect, category, sentiment, opinion)",
    },
    "acosi-extract": {
        "instruction": "Extract aspect-category-sentiment-opinion-implicitIndicator quintuples from input data.",
        "context": "An aspect must be a term existing in input data or null if non-existing; an opinion must be a term existing in input data; the category is one in the predefined list {category_list}; the sentiment is positive, negative or neutral, the implicitIndicator is direct or indirect; do not ask me for more information, I am unable to provide it, and just try your best to finish the task. You can learn from the following examples.",
        "output-format": "(aspect, category, sentiment, opinion, implicitIndicator)",
    },
    "acos-extend": {
        "instruction": "Identify opinion spans from input data and aspect-category-sentiment-opinion quadruples. For implicit opinions (labeled NULL), identify the span of text that best expresses the sentiment implicitly.",
        "context": "An aspect must be a term existing in input data or null if non-existing; an opinion must be a term existing in input data; the category is one in the predefined list; the sentiment is positive, negative or neutral; do not ask me for more information, I am unable to provide it, and just try your best to finish the task. You can learn from the following examples.",
        "output-format": "(opinion)",
    },
}

PROMPTS_OLD = {
    "acos-extract": {
        "instruction": "Given a product review, extract the corresponding ACOS (Aspect-Category-Opinion-Sentiment) quadruples.",
        "context": """Each quadruple is comprised of 4 components:
- Aspect [A]: The span of text in the review that indicates the particular aspect that the customer is referring to. Aspects are not always explicitly stated; if this is the case, use a NULL label for the aspect.
- Category [C]: The category of the aspect, selected from the following list: {category_list}
- Sentiment [S]: The polarity of the sentiment: positive, negative, or neutral.
- Opinion [O]: The span of text in the review that indicates the opinion that expresses the sentiment. Opinions are not always explicitly stated; if this is the case, use a NULL label for the opinion.""",
        "output-format": """[A] aspect [C] category [S] sentiment [O] opinion
[END] is used to mark the end of the set of quadruples associated with a review and [SSEP] is used to separate individual quadruples in the set.""",
        "examples": { 
            "shoes": [
                {
                    "review": "the design is great poor color choices too bland . color choices from previous shoes was much better .",
                    "response": "[A] NULL [C] appearance#form [S] positive [O] design is great [SSEP] [A] NULL [C] appearance#color [S] negative [O] poor color choices [SSEP] [A] shoes [C] appearance#color [S] negative [O] NULL [END]",
                },
                {
                    "review": "omg these are the most comfortable sneakers in the world . i can walk 5 , 6 , 7 miles in them . my whole body may be tired but my feet are great !",
                    "response": "[A] sneakers [C] performance#comfort [S] positive [O] omg these are the most comfortable sneakers in the world [SSEP] [A] NULL [C] performance#use case applicability [S] positive [O] NULL [SSEP] [A] NULL [C] performance#comfort [S] positive [O] NULL [SSEP] [A] NULL [C] performance#general [S] positive [O] NULL [END]",
                }
            ],
            "laptop": [
                {
                    "review": "powers up immediately , great battery life , great keyboard , amazing features .",
                    "response": "[A] powers up [C] laptop#operation_performance [S] positive [O] NULL [SSEP] [A] battery life [C] battery#general [S] positive [O] great [SSEP] [A] keyboard [C] keyboard#general [S] positive [O] NULL [SSEP] [A] NULL [C] laptop#design_features [S] positive [O] amazing [END]"
                },
                {
                    "review": "first one that they shipped was obviously defective , super slow and speakers were garbled .",
                    "response": "[A] NULL [C] shipping#general [S] negative [O] defective [SSEP] [A] NULL [C] shipping#general [S] negative [O] slow [SSEP] [A] speakers [C] multimedia_devices#general [S] negative [O] garbled [END]"
                }
            ],
            "rest": [
                {
                    "review": "the decor is night tho . . . but they really need to clean that vent in the ceiling . . . its quite un - appetizing , and kills your effort to make this place look sleek and modern .",
                    "response": "[A] place [C] ambience#general [S] negative [O] sleek [SSEP] [A] place [C] ambience#general [S] negative [O] modern [SSEP] [A] decor [C] ambience#general [S] positive [O] night [SSEP] [A] vent [C] ambience#general [S] negative [O] un - appetizing [END]"
                },
                {
                    "review": "the food was lousy - too sweet or too salty and the portions tiny .",
                    "response": "[A] food [C] food#quality [S] negative [O] lousy [SSEP] [A] food [C] food#quality [S] negative [O] too sweet [SSEP] [A] food [C] food#quality [S] negative [O] too salty [SSEP] [A] portions [C] food#style_options [S] negative [O] tiny [END]"
                }
            ]
        }
    },
    "acosi-extract": {
        "instruction": "Given a product review, extract the corresponding ACOSI (Aspect-Category-Opinion-Sentiment-Implicit/Explicit) quintuples.",
        "context": """Each quintuple is comprised of 5 components:
- Aspect [A]: The span of text in the review that indicates the particular aspect that the customer is referring to. Aspects are not always explicitly stated; if this is the case, use a NULL label for the aspect.
- Category [C]: The category of the aspect, selected from the following list: {category_list}
- Sentiment [S]: The polarity of the sentiment: positive, negative, or neutral.
- Opinion [O]: The span of text in the review that indicates the opinion that expresses the sentiment. Opinions are not always explicitly stated; if this is the case, please try to identify the span of text that best expresses the sentiment implicitly.
- Implicit Indicator [I]: Indicates whether the opinion is implicit or explicit (indirect or direct).""",
        "output-format": """[A] aspect [C] category [S] sentiment [O] opinion [I] implicit indicator
[END] is used to mark the end of the set of quintuples associated with a review and [SSEP] is used to separate individual quintuples in the set.""",
        "examples": { 
            "shoes": [
                {
                    "review": "the design is great poor color choices too bland . color choices from previous shoes was much better .",
                    "response": "[A] NULL [C] appearance#form [S] positive [O] design is great [I] direct [SSEP] [A] NULL [C] appearance#color [S] negative [O] poor color choices [I] direct [SSEP] [A] shoes [C] appearance#color [S] negative [O] color choices from previous shoes was much better [I] indirect [END]",
                },
                {
                    "review": "omg these are the most comfortable sneakers in the world . i can walk 5 , 6 , 7 miles in them . my whole body may be tired but my feet are great !",
                    "response": "[A] sneakers [C] performance#comfort [S] positive [O] omg these are the most comfortable sneakers in the world [I] direct [SSEP] [A] NULL [C] performance#use case applicability [S] positive [O] i can walk 5 , 6 , 7 miles [I] indirect [SSEP] [A] NULL [C] performance#comfort [S] positive [O] i can walk 5 , 6 , 7 miles [I] indirect [SSEP] [A] NULL [C] performance#general [S] positive [O] my whole body may be tired but my feet are great [I] indirect [END]",
                }
            ],
        }
    },
    "acos-extend": {
        "instruction": "Given a product review and its corresponding ACOS (Aspect-Category-Opinion-Sentiment) quadruples, identify each opinion span. For implicit opinions (labeled NULL), identify the span of text that best expresses the sentiment implicitly.",
        "context": """An online customer review can be organized into a list of ACOS (Aspect-Category-Opinion-Sentiment) quadruples. Each quadruple is comprised of 4 components:
- Aspect [A]: The span of text in the review that indicates the particular aspect that the customer is referring to. Aspects are not always explicitly stated; if this is the case, the aspect is labeled NULL.
- Category [C]: The category of the aspect, selected from a predetermined list.
- Sentiment [S]: The polarity of the sentiment: positive, negative, or neutral.
- Opinion [O]: The span of text in the review that indicates the opinion that expresses the sentiment. Opinions are not always explicitly stated; if this is the case, the opinion is labeled NULL.""",
        "output-format": """[O] opinion
[END] is used to mark the end of the set of quadruples associated with a review and [SSEP] is used to separate individual quadruples in the set.""",
        "examples": { 
            "shoes": [
                {
                    "review": "the design is great poor color choices too bland . color choices from previous shoes was much better .",
                    "acos-quadruples": "[A] NULL [C] appearance#form [S] positive [O] design is great [SSEP] [A] NULL [C] appearance#color [S] negative [O] poor color choices [SSEP] [A] shoes [C] appearance#color [S] negative [O] NULL [END]",
                    "response": "[O] design is great [SSEP] [O] poor color choices [SSEP] [O] color choices from previous shoes was much better [END]",
                    "response-explanation": 'The opinion span for the first quadruple is "design is great", the opinion span for the second quadruple is "poor color choices", and the implicit opinion span for the third quadruple (originally labeled NULL) is "color choices from previous shoes was much better".'
                },
                {
                    "review": "omg these are the most comfortable sneakers in the world . i can walk 5 , 6 , 7 miles in them . my whole body may be tired but my feet are great !",
                    "acos-quadruples": "[A] sneakers [C] performance#comfort [S] positive [O] omg these are the most comfortable sneakers in the world [SSEP] [A] NULL [C] performance#use case applicability [S] positive [O] NULL [SSEP] [A] NULL [C] performance#comfort [S] positive [O] NULL [SSEP] [A] NULL [C] performance#general [S] positive [O] NULL [END]",
                    "response": "[O] omg these are the most comfortable sneakers in the world [SSEP] [O] i can walk 5 , 6 , 7 miles [SSEP] [O] i can walk 5 , 6 , 7 miles [SSEP] [O] my whole body may be tired but my feet are great [END]",
                    "response-explanation": '\n\nThe opinion span for the first quadruple is "omg these are the most comfortable sneakers in the world ", the implicit opinion span for the second quadruple (originally labeled NULL) is "i can walk 5 , 6 , 7 miles", the opinion span for the third quadruple (originally labeled NULL) is "i can walk 5 , 6 , 7 miles", and the opinion span for the fourth quadruple (originally labeled NULL) is "my whole body may be tired but my feet are great".'
                }
            ],
        }
    },
}
