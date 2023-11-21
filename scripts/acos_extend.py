from pickle import dump, load
from pprint import pprint
from pipeline import run_pipeline
from utils import get_args, format_output


def get_ACOS_extend_prompt():
    prompt = """
    Given an online customer review and its corresponding ACOS quadruples, 
    convert the ACOS (Aspect-Category-Opinion-Sentiment) quadruples to ACOSI (Aspect-Category-Opinion-Sentiment-Implicit/Explicit) quintuples by\n
    1) identifying and labeling the corresponding opinion span for implicit opinions (labeled NULL in ACOS).\n
    2) adding an the implicit indicator term to each quadruple (making it a quintuple) that indicates whether the opinion is implicit or explicit (indirect/direct).\n\n
    """

    response_head = "ACOSI quintuples:"

    example1 = f"""
    Example 1:\n\n

    Review: looks nice , and the surface is smooth , but certain apps take seconds to respond .\n
    ACOS quadruples: [(Aspect: "surface", Category: design, Sentiment: Positive, Opinion: "smooth"), 
                      (Aspect: NULL, Category: design, Sentiment: Positive, Opinion: "nice"), 
                      (Aspect: "apps", Category: software, Sentiment: Negative, Opinion: NULL)]
                      \n

    Response:\n
    {response_head} [(Aspect: "surface", Category: design, Sentiment: Positive, Opinion: "smooth", Implicit/Explicit: direct),
                       (Aspect: NULL, Category: design, Sentiment: Positive, Opinion: "nice", Implicit/Explicit: direct),
                       (Aspect: "apps", Category: software, Sentiment: Negative, Opinion: "apps take seconds to respond", Implicit/Explicit: indirect)]
                       \n\n
    """

    example2 = f"""
    Example 2:\n\n

    Review: with the theater 2 blocks away we had a delicious meal in a beautiful room .\n
    ACOS quadruples: [(Aspect: "meal", Category: food#quality, Sentiment: Positive, Opinion: "delicious"), 
                      (Aspect: "room", Category: ambience#general, Sentiment: Positive, Opinion: "beautiful"), 
                      (Aspect: NULL, Category: location#general, Sentiment: Positive, Opinion: NULL)]
                      \n
    
    Response:\n
    {response_head} [(Aspect: "meal", Category: food#quality, Sentiment: Positive, Opinion: "delicious", Implicit/Explicit: direct),
                       (Aspect: "room", Category: ambience#general, Sentiment: Positive, Opinion: "beautiful", Implicit/Explicit: direct),
                       (Aspect: NULL, Category: location#general, Sentiment: Positive, Opinion: "theater 2 blocks away", Implicit/Explicit: indirect)]
                       \n\n
    """

    examples = [example1, example2]

    return prompt, examples, response_head


def main(args):
    prompt, examples, response_head = get_ACOS_extend_prompt()
    output, response_key = run_pipeline(args, prompt, examples, absa_task="acos-extend")
    # with open(args.output_file, "r") as f:
    #     output = load(f)
    # response_key = "#### Response:"
    pprint(output)
    try:
        formatted_output = format_output(output, response_key, response_head)
        dump(formatted_output, args.output_file)
    except Exception as e:
        dump(output, args.output_file)
        print(e)


if __name__ == "__main__":
    args = get_args()
    main(args)
