prompts = {
    "acos_extend": [
        {
            "prompt": """
                Given an online customer review and its corresponding ACOS (Aspect-Category-Opinion-Sentiment) quadruples,  
                identify and label the corresponding opinion span for implicit opinions (labeled NULL in ACOS).
                """,
            "examples": [
                """
                Example 1:\n\n

                Review: looks nice , and the surface is smooth , but certain apps take seconds to respond .\n
                ACOS quadruples: [(Aspect: "surface", Category: design, Sentiment: positive, Opinion: "smooth"), 
                                  (Aspect: NULL, Category: design, Sentiment: positive, Opinion: "nice"), 
                                  (Aspect: "apps", Category: software, Sentiment: negative, Opinion: NULL)]
                
                Response:\n
                {response_head} [Opinion: "smooth", Opinion: "nice", Implicit Opinion: "apps take seconds to respond"]
                """,
                """
                Example 2:\n\n

                Review: with the theater 2 blocks away we had a delicious meal in a beautiful room .\n
                ACOS quadruples: [(Aspect: "meal", Category: food#quality, Sentiment: positive, Opinion: "delicious"),
                                  (Aspect: NULL, Category: location#general, Sentiment: positive, Opinion: NULL),
                                  (Aspect: "room", Category: ambience#general, Sentiment: positive, Opinion: "beautiful")]
                                \n
                
                Response:\n
                {response_head} [Opinion: "delicious", Implicit Opinion: "theater 2 blocks away", Opinion: "beautiful"]
                                \n\n
                """,
            ],
        },
    ],
    "acos_extract": [
        {
            "prompt": """    
            Given a online customer review, 
            extract the corresponding ACOS (Aspect-Category-Opinion-Sentiment) quadruples. \n
            
            Each quadruple is comprised of 4 components:\n
            - Aspect: The span of text in the review that indicates the particular aspect that the customer is referring to. 
                    Aspects are not always explicitly stated; if this is the case, use a NULL label for the aspect.\n
            - Category: The category of the aspect, selected from the following list: {category_list}\n
            - Sentiment: The polarity of the sentiment: positive, negative, or neutral.\n
            - Opinion: The span of text in the review that indicates the opinion that expresses the sentiment.
                    Opinions are not always explicitly stated; if this is the case, use a NULL lable for the opinion.\n\n""",
            "examples": [
                """
                Example 1:\n\n

                Review: the food was lousy - too sweet or too salty and the portions tiny .\n

                Response:\n
                {response_head} [(Aspect: "food", Category: food#quality, Sentiment: negative, Opinion: "lousy"),
                                (Aspect: "food", Category: food#quality, Sentiment: negative, Opinion: "too sweet"),
                                (Aspect: "food", Category: food#quality, Sentiment: negative, Opinion: "too salty"),
                                (Aspect: "portions", Category: food#style_options, Sentiment: negative, Opinion: "tiny")]
                                \n\n
                """,
                """
                Example 1:\n\n

                Review: the decor is night tho . . . but they really need to clean that vent in the ceiling . . . its quite un - appetizing , and kills your effort to make this place look sleek and modern .\n
                
                Response:\n
                {response_head} [(Aspect: "place", Category: ambience#general, Sentiment: negative, Opinion: "sleek"),
                                (Aspect: "place", Category: ambience#general, Sentiment: negative, Opinion: "modern"),
                                (Aspect: "decor", Category: ambience#general, Sentiment: positive, Opinion: "night"),
                                (Aspect: "vent", Category: ambience#general, Sentiment: negative, Opinion: "un - appetizing")]
                                \n\n
                """,
                """
                Example 2:\n\n

                Review: powers up immediately , great battery life , great keyboard , amazing features .\n

                Response:\n
                {response_head} [(Aspect: "powers up", Category: laptop#operation_performance, Sentiment: positive, Opinion: NULL), 
                                (Aspect: "battery life", Category: battery#general, Sentiment: positive, Opinion: "great"), 
                                (Aspect: "keyboard", Category: keyboard#general, Sentiment: positive, Opinion: NULL),
                                (Aspect: NULL, Category: laptop#design_features, Sentiment: positive, Opinion: "amazing")]
                                \n\n
                """,
            ],
        }
    ],
    "acosi_extract": [
        {
            "prompt": """
                Given a product review, 
                extract the corresponding ACOSI (Aspect-Category-Opinion-Sentiment-Implicit/Explicit) quintuples.\n

                Each quintuple is comprised of 5 components:\n
                - Aspect: The span of text in the review that indicates the particular aspect that the customer is referring to. 
                        Aspects are not always explicitly stated; if this is the case, use a NULL label for the aspect.\n
                - Category: The category of the aspect, selected from the following list: {category_list}\n
                - Sentiment: The polarity of the sentiment: positive, negative, or neutral.\n
                - Opinion: The span of text in the review that indicates the opinion that expresses the sentiment.
                        Opinions are not always explicitly stated; if this is the case, please try to identify the span of text that best expresses the sentiment implicitly.\n
                - Implicit Indicator: Indicates whether the opinion is implicit or explicit (indirect or direct).
                \n\n
                """,
            "examples": [
                """
                Example 1:\n\n

                Review: the design is great poor color choices too bland . color choices from previous shoes was much better . \n

                Response: \n
                {response_head} [(Aspect: NULL, Category: appearance#form, Sentiment: positive, Opinion: "design is great", Implicit/Explicit: direct), 
                                (Aspect: NULL, Category: appearance#color, Sentiment: negative, Opinion: "poor color choices", Implicit/Explicit: direct), 
                                (Aspect: "shoes", Category: appearance#color, Sentiment: negative, Opinion: "color choices from previous shoes was much better", Implicit/Explicit: indirect)]
                                \n\n

                """,
                """
                Example 2: \n\n

                Review: had to order a larger size than what i normally wear . shoe would be better if offered as an adjustable shoe . shoe is overpriced for quality . i bought cheaper slides in the past that were more comfortable . \n

                Response: \n
                {response_head} [(Aspect: NULL, Category: performance#sizing_fit, Sentiment: neutral, Opinion: "had to order a larger size than what i normally wear", Implicit/Explicit: direct), 
                                (Aspect: NULL, Category: contextofuse#purchase\\\\_context, Sentiment: negative, Opinion: "had to order a larger size than what i normally wear", Implicit/Explicit: direct), 
                                (Aspect: "shoe", Category: appearance#form, Sentiment: neutral, Opinion: "would be better if offered as an adjustable shoe", Implicit/Explicit: direct), 
                                (Aspect: "shoe", Category: cost/value, Sentiment: negative, Opinion: "overpriced for quality", Implicit/Explicit: direct), 
                                (Aspect: "slides", Category: cost/value, Sentiment: negative, Opinion: "i bought cheaper slides in the past that were more comfortable", Implicit/Explicit: direct), 
                                (Aspect: "slides", Category: performance#comfort, Sentiment: negative, Opinion: "i bought cheaper slides in the past that were more comfortable", Implicit/Explicit: direct)]
                                \n\n
                """,
            ],
        }
    ],
}
