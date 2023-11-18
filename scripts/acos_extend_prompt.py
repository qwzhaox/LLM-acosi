from argparse import ArgumentParser
from pipeline import run_pipeline
from pickle import dump
from ast import eval

def get_ACOS_extend_prompt():
    prompt = """
    Given an online customer review and its corresponding ACOS quadruples, 
    convert the ACOS (Aspect-Category-Opinion-Sentiment) quadruples to ACOSI (Aspect-Category-Opinion-Sentiment-Implicit/Explicit) quintuples by\n
    1) identifying and labeling the corresponding opinion span for implicit opinions (labeled NULL in ACOS).\n
    2) adding an the implicit indicator term to each quadruple (making it a quintuple) that indicates whether the opinion is implicit or explicit (indirect/direct).\n\n
    """

    example1 = """
    Example 1:\n\n

    Review: looks nice , and the surface is smooth , but certain apps take seconds to respond .\n
    ACOS quadruples: [(Aspect: "surface", Category: design, Sentiment: Positive, Opinion: "smooth"), 
                      (Aspect: NULL, Category: design, Sentiment: Positive, Opinion: "nice"), 
                      (Aspect: "apps", Category: software, Sentiment: Negative, Opinion: NULL)]
                      \n

    Response:\n
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
    
    Response:\n
    ACOSI quintuples: [(Aspect: "meal", Category: food#quality, Sentiment: Positive, Opinion: "delicious", Implicit/Explicit: direct),
                       (Aspect: "room", Category: ambience#general, Sentiment: Positive, Opinion: "beautiful", Implicit/Explicit: direct),
                       (Aspect: NULL, Category: location#general, Sentiment: Positive, Opinion: "theater 2 blocks away", Implicit/Explicit: indirect)]
                       \n\n
    """

    examples = [example1, example2]

    return prompt, examples
    

def remove_tags(text):
    return text.replace("Aspect: ", "").replace("Categroy: ", "").replace("Sentiment: ", "").replace("Opinion: ", "").replace("Implicit/Explicit: ", "")


def add_quotations(text):
    return text.replace("\"", "").replace("\'", "").replace("(", "(\'").replace(")", "\')").replace(", ", ",").replace(",", "\',\'")


def format_output(output, response_key):
    formatted_output = []
    for out in output:
        prediction = out["generated_text"].strip()
        if response_key in prediction:
            prediction = prediction.split(response_key)[1].strip()

        prediction = remove_tags(prediction)
        prediction = add_quotations(prediction)
        prediction = prediction.lower()

        formatted_tuple = eval(prediction)
        formatted_output.append(formatted_tuple)
    
    return formatted_output


def main(args):
    prompt, examples = get_ACOS_extend_prompt()
    output, response_key = run_pipeline(args, prompt, examples, absa_task="extend")
    formatted_output = format_output(output, response_key)
    with open(args.output_file, "w") as f:
        dump(formatted_output, f)


if "__name__" == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="LLM model name")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer name")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--remote", action="store_true", help="Whether to trust remote code")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max new tokens")
    parser.add_argument("--dataset_file", type=str, required=True, help="Dataset file")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    args = parser.parse_args()

    main(args)
        
