def get_ACOS_extend_prompt():
    prompt = """
    Given a product review, 
    extract the ACOS (Aspect-Category-Opinion-Sentiment) quadruples by
    1) identifying and labeling the corresponding aspect, category, opinion, and sentiment spans.
    2) give the opinion annotaion a value of "NULL" if no explicit opinion span can be found.
    Note that, we will give you product reviews of either restaurants or laptops, so in your response, the category can and can only fall into one of the followings:
    1) for restaurants
    [
    "location#general",
    "food#prices",
    "food#quality",
    "food#general",
    "ambience#general",
    "service#general",
    "restaurant#prices",
    "drinks#prices",
    "restaurant#miscellaneous",
    "drinks#quality",
    "drinks#style_options",
    "restaurant#general",
    "food#style_options"
    ]
    2) for laptops:
    [
    "keyboard#operation_performance",
    "os#operation_performance",
    "out_of_scope#operation_performance",
    "ports#general",
    "optical_drives#general",
    "laptop#operation_performance",
    "optical_drives#operation_performance",
    "optical_drives#usability",
    "multimedia_devices#general",
    "keyboard#general",
    "os#miscellaneous",
    "software#operation_performance",
    "display#operation_performance",
    "shipping#quality",
    "hard_disc#quality",
    "motherboard#general",
    "graphics#general",
    "multimedia_devices#connectivity",
    "display#general",
    "memory#operation_performance",
    "os#design_features",
    "out_of_scope#usability",
    "software#design_features",
    "graphics#design_features",
    "ports#connectivity",
    "support#design_features",
    "display#quality",
    "software#price",
    "shipping#general",
    "graphics#operation_performance",
    "hard_disc#miscellaneous",
    "display#design_features",
    "cpu#operation_performance",
    "mouse#general",
    "keyboard#portability",
    "hardware#price",
    "support#quality",
    "hardware#quality",
    "motherboard#operation_performance",
    "multimedia_devices#quality",
    "battery#design_features",
    "mouse#usability",
    "os#price",
    "shipping#operation_performance",
    "laptop#quality",
    "laptop#portability",
    "fans&cooling#general",
    "battery#general",
    "os#usability",
    "hardware#usability",
    "optical_drives#design_features",
    "fans&cooling#operation_performance",
    "memory#general",
    "company#general",
    "power_supply#general",
    "hardware#general",
    "mouse#design_features",
    "software#general",
    "keyboard#quality",
    "power_supply#quality",
    "software#quality",
    "multimedia_devices#usability",
    "power_supply#connectivity",
    "multimedia_devices#price",
    "multimedia_devices#operation_performance",
    "ports#design_features",
    "hardware#operation_performance",
    "shipping#price",
    "hardware#design_features",
    "memory#usability",
    "cpu#quality",
    "ports#quality",
    "ports#portability",
    "motherboard#quality",
    "display#price",
    "os#quality",
    "graphics#usability",
    "cpu#design_features",
    "hard_disc#general",
    "hard_disc#operation_performance",
    "battery#quality",
    "laptop#usability",
    "company#design_features",
    "company#operation_performance",
    "support#general",
    "fans&cooling#quality",
    "memory#design_features",
    "ports#usability",
    "hard_disc#design_features",
    "power_supply#design_features",
    "keyboard#miscellaneous",
    "laptop#miscellaneous",
    "keyboard#usability",
    "cpu#price",
    "laptop#design_features",
    "keyboard#price",
    "warranty#quality",
    "display#usability",
    "support#price",
    "cpu#general",
    "out_of_scope#design_features",
    "out_of_scope#general",
    "software#usability",
    "laptop#general",
    "warranty#general",
    "company#price",
    "ports#operation_performance",
    "power_supply#operation_performance",
    "keyboard#design_features",
    "support#operation_performance",
    "hard_disc#usability",
    "os#general",
    "company#quality",
    "memory#quality",
    "software#portability",
    "fans&cooling#design_features",
    "multimedia_devices#design_features",
    "laptop#connectivity",
    "battery#operation_performance",
    "hard_disc#price",
    "laptop#price"
    ]
    """

    example1 = """
    Example 1:\n\n

    Review: the food was lousy - too sweet or too salty and the portions tiny .\n
    ACOS quadruples: [(Aspect: "food", Category: food#quality, Sentiment: Negative, Opinion: "lousy"),
                      (Aspect: "food", Category: food#quality, Sentiment: Negative, Opinion: "too sweet")
                      (Aspect: "food", Category: food#quality, Sentiment: Negative, Opinion: "too salty"),
                      (Aspect: "portions", Category: food#style_options, Sentiment: Negative, Opinion: "tiny")]
                      \n
    """

    example2 = """
    Example 1:\n\n

    Review: first one that they shipped was obviously defective , super slow and speakers were garbled .\n
    ACOS quadruples: [(Aspect: NULL, Category: shipping#general, Sentiment: Negative, Opinion: "defective"), 
                      (Aspect: NULL, Category: shipping#general, Sentiment: Negative, Opinion: "slow"), 
                      (Aspect: "speakers", Category: multimedia_devices#general, Sentiment: Negative, Opinion: "garbled")]
                      \n
    """

    examples = [example1, example2]

    return prompt, examples
