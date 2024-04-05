from argparse import ArgumentParser
from json import load, dump

parser = ArgumentParser()
parser.add_argument("--input_file", required=True)
parser.add_argument("--output_file", required=True)

args = parser.parse_args()

with open(args.input_file, "r") as file:
    score_dict = load(file)

revs_and_annots = score_dict["reviews"]
predictions = [rev_and_annot["pred"] for rev_and_annot in revs_and_annots]

with open(args.output_file, "w") as file:
    dump(predictions, file, indent=4)