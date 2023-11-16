def get_ACOSI_extract_prompt():
    prompt = """
    Given a product review, extract ACOSI (Aspect-Category-Opinion-Sentiment-Implicit/Explicit) quintuples by identifying and labeling the aspect, category, sentiment, opinion, and whether the opinion is implicit or explicit (indirect or direct).
    You will be working with reviews of shoes, so the category for an extracted quintuple can only fall into one of the following:
    
    {
        "appearance color",
        "appearance form",
        "appearance general",
        "appearance material",
        "appearance misc",
        "appearance shoe component",
        "contextofuse place",
        "contextofuse purchase_context",
        "contextofuse review_temporality",
        "contextofuse usage frequency",
        "contextofuse use case",
        "cost_value",
        "general",
        "misc",
        "performance comfort",
        "performance durability",
        "performance general",
        "performance misc",
        "performance sizing_fit",
        "performance support_stability",
        "performance use case applicability",
        "versatility"
    }
    """

    example1 = """
    Example 1:\n\n

    Review: the design is great poor color choices too bland . color choices from previous shoes was much better . \n

    ACOSI quintuples: [(Aspect: 'null', Category: 'appearance form', Sentiment: 'positive', Opinion: 'design is great', Implicit/Explicit: 'direct'), 
                        (Aspect: 'null', Category: 'appearance color', Sentiment: 'negative', Opinion: 'poor color choices', Implicit/Explicit: 'direct'), 
                        (Aspect: 'shoes', Category: 'appearance color', Sentiment: 'negative', Opinion: 'color choices from previous shoes was much better', Implicit/Explicit: 'indirect')]
                        \n

    """

    example2 = """
    Example 2: \n\n

    Review: had to order a larger size than what i normally wear . shoe would be better if offered as an adjustable shoe . shoe is overpriced for quality . i bought cheaper slides in the past that were more comfortable . \n

    ACOSI quintuples: [('Aspect: null', Category: 'performance sizing_fit', Sentiment: 'neutral', Opinion: 'had to order a larger size than what i normally wear', Implicit/Explicit: 'direct'), 
                        ('Aspect: null', Category: 'contextofuse purchase_context', Sentiment: 'negative', Opinion: 'had to order a larger size than what i normally wear', Implicit/Explicit: 'direct'), 
                        ('Aspect: shoe', Category: 'appearance form', Sentiment: 'neutral', Opinion: 'would be better if offered as an adjustable shoe', Implicit/Explicit: 'direct'), 
                        ('Aspect: shoe', Category: 'cost_value', Sentiment: 'negative', Opinion: 'overpriced for quality', Implicit/Explicit: 'direct'), 
                        ('Aspect: slides', Category: 'cost_value', Sentiment: 'negative', Opinion: 'i bought cheaper slides in the past that were more comfortable', Implicit/Explicit: 'direct'), 
                        ('Aspect: slides', Category: 'performance comfort', Sentiment: 'negative', Opinion: 'i bought cheaper slides in the past that were more comfortable', Implicit/Explicit: 'direct')]
                        \n
    """

    examples = [example1, example2]

    return prompt, examples
