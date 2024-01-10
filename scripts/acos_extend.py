from pipeline import get_model_output
from utils import (
    get_args,
    format_output,
    get_formatted_output_w_metadata,
    dump_output,
)

OPINION_IDX = 3


def get_ACOS_extend_prompt():
    prompt = """An online customer review can be organized into a list of ACOS (Aspect-Category-Opinion-Sentiment) quadruples.
    
Each quadruple is comprised of 4 components:
- Aspect [A]: The span of text in the review that indicates the particular aspect that the customer is referring to. Aspects are not always explicitly stated; if this is the case, the aspect is labeled NULL.
- Category [C]: The category of the aspect, selected from a predetermined list.
- Sentiment [S]: The polarity of the sentiment: positive, negative, or neutral.
- Opinion [O]: The span of text in the review that indicates the opinion that expresses the sentiment. Opinions are not always explicitly stated; if this is the case, the opinion is labeled NULL.

Given an online customer review and its corresponding ACOS (Aspect-Category-Opinion-Sentiment) quadruples, identify each opinion span. For implicit opinions (labeled NULL), identify the span of text that best expresses the sentiment implicitly.
[END] is used to mark the end of the set of quadruples associated with a review and [SSEP] is used to separate individual quadruples in the set.\n
"""

    response_head = "Opinion spans:"

    # example1 = [
    #     "Review: looks nice , and the surface is smooth , but certain apps take seconds to respond .\nACOS quadruples: [A] surface [C] design [S] positive [O] smooth [SSEP] [A] NULL [C] design [S] positive [O] nice [SSEP] [A] apps [C] software [S] negative [O] NULL [END]\n",
    #     f'{response_head} [O] smooth [SSEP] [O] nice [SSEP] [O] apps take seconds to respond [END]\n\nThe opinion span for the first quadruple is "smooth", the opinion span for the second quadruple is "nice", and the implicit opinion span for the third quadruple (originally labeled NULL) is "apps take seconds to respond".\n\n',
    # ]

    # example2 = [
    #     "Review: with the theater 2 blocks away we had a delicious meal in a beautiful room .\nACOS quadruples: [A] meal [C] food#quality [S] positive [O] delicious [SSEP] [A] NULL [C] location#general [S] positive [O] NULL [SSEP] [A] room [C] ambience#general [S] positive [O] beautiful [END]\n",
    #     f'{response_head} [O] delicious [SSEP] [O] theater 2 blocks away [SSEP] [O] beautiful [END]\n\nThe opinion span for the first quadruple is "delicious", the implicit opinion span for the second quadruple (originally labeled NULL) is "theater 2 blocks away", and the opinion span for the third quadruple is "beautiful".\n\n',
    # ]

    example1 = [
        "Review: the design is great poor color choices too bland . color choices from previous shoes was much better .\nACOS quadruples: [A] NULL [C] appearance#form [S] positive [O] design is great [SSEP] [A] NULL [C] appearance#color [S] negative [O] poor color choices [SSEP] [A] shoes [C] appearance#color [S] negative [O] NULL [END]\n",
        f'{response_head} [O] design is great [SSEP] [O] poor color choices [SSEP] [O] color choices from previous shoes was much better [END]\n\nThe opinion span for the first quadruple is "design is great", the opinion span for the second quadruple is "poor color choices", and the implicit opinion span for the third quadruple (originally labeled NULL) is "color choices from previous shoes was much better".\n\n',
    ]

    example2 = [
        "Review: omg these are the most comfortable sneakers in the world . i can walk 5 , 6 , 7 miles in them . my whole body may be tired but my feet are great !\nACOS quadruples: [A] sneakers [C] performance#comfort [S] positive [O] omg these are the most comfortable sneakers in the world [SSEP] [A] NULL [C] performance#use case applicability [S] positive [O] NULL [SSEP] [A] NULL [C] performance#comfort [S] positive [O] NULL [SSEP] [A] NULL [C] performance#general [S] positive [O] NULL [END]\n",
        f'{response_head} [O] omg these are the most comfortable sneakers in the world [SSEP] [O] i can walk 5 , 6 , 7 miles [SSEP] [O] i can walk 5 , 6 , 7 miles [SSEP] [O] my whole body may be tired but my feet are great [END]\n\nThe opinion span for the first quadruple is "omg these are the most comfortable sneakers in the world ", the implicit opinion span for the second quadruple (originally labeled NULL) is "i can walk 5 , 6 , 7 miles", the opinion span for the third quadruple (originally labeled NULL) is "i can walk 5 , 6 , 7 miles", and the opinion span for the fourth quadruple (originally labeled NULL) is "my whole body may be tired but my feet are great".\n\n',
    ]

    examples = [example1, example2]

    return prompt, examples, response_head


def get_ACOS_annotations(len_formatted_output):
    with open(args.dataset_file, "r") as f:
        dataset = f.readlines()

    assert len(dataset) == len_formatted_output

    acos_annotations = [eval(x.split("####")[1]) for x in dataset]

    return acos_annotations


def get_ACOSI_annotations(acos_annotations, formatted_output):
    acosi_annotations = []
    for quadruples, opinion_only_quints in zip(acos_annotations, formatted_output):
        cur_acosi_annotation = []
        if len(quadruples) == len(opinion_only_quints):
            for quad, opinion_only_quint in zip(quadruples, opinion_only_quints):
                if quad[OPINION_IDX].lower() == "null":
                    quad[OPINION_IDX] = opinion_only_quint[OPINION_IDX]
                    quad.append("indirect")
                else:
                    quad.append("direct")
                quint = tuple(quad)
                cur_acosi_annotation.append(quint)
        else:
            for quad in quadruples:
                if quad[OPINION_IDX].lower() == "null":
                    quad.append("indirect")
                else:
                    quad.append("direct")
                quint = tuple(quad)
                cur_acosi_annotation.append(quint)

        acosi_annotations.append(cur_acosi_annotation)
    return acosi_annotations


def main(args):
    prompt, examples, response_head = get_ACOS_extend_prompt()
    opinion_spans, response_key, reviews = get_model_output(
        args, prompt, examples, absa_task="acos-extend"
    )
    formatted_output, raw_predictions = format_output(
        opinion_spans, response_key, response_head
    )
    acos_annotations = get_ACOS_annotations(len(formatted_output))
    formatted_output = get_ACOSI_annotations(acos_annotations, formatted_output)

    formatted_output_w_metadata = get_formatted_output_w_metadata(
        formatted_output, raw_predictions, reviews
    )
    dump_output(args.output_file, formatted_output)
    dump_output(args.output_file + "_METADATA", formatted_output_w_metadata)


if __name__ == "__main__":
    args = get_args()
    main(args)
