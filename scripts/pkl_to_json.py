from pickle import load
from json import dump
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-p", "--pkl_file", type=str, required=True, help="Pickle file")
parser.add_argument("-j", "--json_file", type=str, required=True, help="JSON file")
args = parser.parse_args()

with open(args.pkl_file, "rb") as f:
    data = load(f)

with open(args.json_file, "w") as f:
    dump(data, f)
