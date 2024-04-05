from dotenv import load_dotenv
from pipeline import Pipeline, load
from utils import (
    get_args,
    format_output,
    get_formatted_output_and_metadata,
    dump_output,
    ENV_PATH
)

OPINION_IDX = 3


def get_ACOS_annotations(annotation_source, len_formatted_output):
    if annotation_source == "true":
        with open(args.dataset_file, "r") as f:
            dataset = f.readlines()
        assert len(dataset) == len_formatted_output
        acos_annotations = [eval(x.split("####")[1]) for x in dataset]
    elif ("mvp" in annotation_source) or (annotation_source == "gen-scl-nat"):
        with open(f"model_output/supervised/{annotation_source}/pred.json", "r") as file:
            acos_annotations = load(file)
            assert len(acos_annotations) == len_formatted_output
    else:
        raise NotImplementedError("Invalid annotation source.")

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
    load_dotenv(ENV_PATH)

    pipeline = Pipeline(args)
    output, reviews = pipeline.get_model_output()
    
    formatted_output, raw_predictions = format_output(output, is_old_prompt=args.is_old_prompt, is_combo_prompt=args.is_combo_prompt)

    if args.absa_task == "acos-extract":
        formatted_output = [[quint[:-1] for quint in quints] for quints in formatted_output]
    elif args.absa_task == "acos-extend":
        acos_annotations = get_ACOS_annotations(args.annotation_source, len_formatted_output=len(formatted_output))
        formatted_output = get_ACOSI_annotations(acos_annotations, formatted_output)

    formatted_output_and_metadata = get_formatted_output_and_metadata(
        formatted_output, raw_predictions, reviews
    )
    dump_output(args.output_file, formatted_output)
    dump_output(args.output_file + "_METADATA", formatted_output_and_metadata)


if __name__ == "__main__":
    args = get_args()
    main(args)