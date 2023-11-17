def get_ACOS_extend_prompt():
    prompt = """
    Given an online customer review and its corresponding ACOS quadruples, 
    convert the ACOS (Aspect-Category-Opinion-Sentiment) quadruples to ACOSI (Aspect-Category-Opinion-Sentiment-Implicit/Explicit) quintuples by\n
    1) identifying and labeling the corresponding opinion span for implicit opinions (labeled NULL in ACOS).\n
    2) adding an the implicit indicator term to each quadruple (making it a quintuple) that indicates whether the opinion is implicit or explicit.\n\n
    """

    example1 = """
    Example 1:\n\n

    Review: looks nice , and the surface is smooth , but certain apps take seconds to respond .\n
    ACOS quadruples: [(Aspect: "surface", Category: design, Sentiment: Positive, Opinion: "smooth"), 
                      (Aspect: NULL, Category: design, Sentiment: Positive, Opinion: "nice"), 
                      (Aspect: "apps", Category: software, Sentiment: Negative, Opinion: NULL)]
                      \n

    Your Response: \n
    ACOSI quintuples: [(Aspect: "surface", Category: design, Sentiment: Positive, Opinion: "smooth", Implicit/Explicit: direct),
                       (Aspect: NULL, Category: design, Sentiment: Positive, Opinion: "nice", Implicit/Explicit: direct),
                       (Aspect: "apps", Category: software, Sentiment: Negative, Opinion: "apps take seconds to respond", Implicit/Explicit: indirect)]
                       \n\n
    """

    example2 = """
    Example 2:\n\n

    Review: with the theater 2 blocks away we had a delicious meal in a beautiful room .\n
    ACOS quadruples: [(Aspect: "meal", Category: food#quality, Sentiment: Positive, Opinion: "delicious"), 
                      (Aspect: "room", Category: ambience#general, Sentiment: Positive, Opinion: "beautiful"), 
                      (Aspect: NULL, Category: location#general, Sentiment: Positive, Opinion: NULL)]
                      \n
    
    Your Response: \n
    ACOSI quintuples: [(Aspect: "meal", Category: food#quality, Sentiment: Positive, Opinion: "delicious", Implicit/Explicit: direct),
                       (Aspect: "room", Category: ambience#general, Sentiment: Positive, Opinion: "beautiful", Implicit/Explicit: direct),
                       (Aspect: NULL, Category: location#general, Sentiment: Positive, Opinion: "theater 2 blocks away", Implicit/Explicit: indirect)]
                       \n\n
    """

    examples = [example1, example2]

    return prompt, examples
