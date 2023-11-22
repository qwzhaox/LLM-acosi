from pickle import dump, load
from pprint import pprint
from pipeline import run_pipeline
from utils import get_args, format_output

OPINION_IDX = 3


def get_ACOS_extend_prompt():
    prompt = """
    Given an online customer review and its corresponding ACOS (Aspect-Category-Opinion-Sentimeent) quadruples, 
    identify and label the corresponding opinion span for implicit opinions (labeled NULL in ACOS).\n\n
    """

    response_head = "Opinion spans:"

    example1 = f"""
    Example 1:\n\n

    Review: looks nice , and the surface is smooth , but certain apps take seconds to respond .\n
    ACOS quadruples: [(Aspect: "surface", Category: design, Sentiment: positive, Opinion: "smooth"), 
                      (Aspect: NULL, Category: design, Sentiment: positive, Opinion: "nice"), 
                      (Aspect: "apps", Category: software, Sentiment: negative, Opinion: NULL)]
                      \n

    Response:\n
    {response_head} [Opinion: "smooth", Opinion: "nice", Implicit Opinion: "apps take seconds to respond"]
                       \n\n
    """

    example2 = f"""
    Example 2:\n\n

    Review: with the theater 2 blocks away we had a delicious meal in a beautiful room .\n
    ACOS quadruples: [(Aspect: "meal", Category: food#quality, Sentiment: positive, Opinion: "delicious"),
                      (Aspect: NULL, Category: location#general, Sentiment: positive, Opinion: NULL),
                      (Aspect: "room", Category: ambience#general, Sentiment: positive, Opinion: "beautiful")]
                      \n
    
    Response:\n
    {response_head} [Opinion: "delicious", Implicit Opinion: "theater 2 blocks away", Opinion: "beautiful"]
                       \n\n
    """

    examples = [example1, example2]

    return prompt, examples, response_head


def main(args):
    prompt, examples, response_head = get_ACOS_extend_prompt()
    opinion_spans, response_key = run_pipeline(
        args, prompt, examples, absa_task="acos-extend"
    )
    # with open(args.output_file, "r") as f:
    #     output = load(f)
    # response_key = "#### Response:"
    formatted_output = format_output(opinion_spans, response_key, response_head)
    pprint(formatted_output)

    with open(args.dataset_file, "r") as f:
        dataset = f.readlines()

    assert len(dataset) == len(formatted_output)

    acos_annotations = [eval(x.split("####")[1]) for x in dataset]
    acosi_annotations = []

    for quadruples, opinion_spans in zip(acos_annotations, formatted_output):
        cur_acosi_annotation = []
        if len(quadruples) == len(opinion_spans):
            for quad, opinion_span in zip(quadruples, opinion_spans):
                if quad[OPINION_IDX] == "NULL":
                    quad[OPINION_IDX] = opinion_span
                    quint = tuple(quad.append("indirect"))
                else:
                    quint = tuple(quad.append("direct"))
                cur_acosi_annotation.append(quint)

        acosi_annotations.append(cur_acosi_annotation)

    pprint(acosi_annotations)

    with open(args.output_file, "wb") as f:
        dump(formatted_output, f)


if __name__ == "__main__":
    args = get_args()
    main(args)
