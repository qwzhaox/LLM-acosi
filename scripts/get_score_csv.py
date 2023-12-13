import json
import csv
import subprocess
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--absa_task",
    help="absa task name, enter acos or acosi",
    required=True
)
parser.add_argument(
    "--dataset",
    help="dataset name, enter laptop, rest, or shoes",
    required=True
)

# get all score.json files
score_path="~/Shoes-ACOSI-models/EECS595Project/data/eval_output/meta-llama/"
command = "find " + score_path + " -name score.json"
score_files = subprocess.run(command, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
score_files = score_files.stdout.strip().split('\n')

args = parser.parse_args()

# define csv file fields by task annotations
task = args.absa_task
dataset = args.dataset
field = ["precision", "recall", "f1-score", "global IoU", "avg local IoU"]
annotation = []
model_scores = {}
if task == "acos":
    annotation = ["exact", "aspect", "category", "opinion", "sentiment"]
    model_scores = {"exact": {}, "aspect": {}, "category": {}, "opinion": {}, "sentiment": {}}
elif task == "acosi":
    annotation = ["exact", "aspect", "category", "opinion", "sentiment", "implicit indicator"]
    model_scores = {"exact": {}, "aspect": {}, "category": {}, "opinion": {}, "sentiment": {}, "implicit indicator": {}}
else:
    print("invalid task")
    exit


# get all scores
for file in score_files:
    file_arr = file.split('/')
    model = file_arr[8]
    curr_task = file_arr[9].split('-')[0]
    curr_dataset = file_arr[10]

    if not curr_task == task:
        print("now performing " + task + ", skip the current file on " + curr_task)
        continue
    if not curr_dataset == dataset:
        print("now performing " + dataset + ", skip the current file on " + curr_dataset)
    

    # get scores for each annotation from all model scores
    data = {}
    with open(file, "r") as f:
        data = json.load(f)
        f.close()
    for a in annotation:
        model_scores[a][model] = data[a]

# print(model_scores)
# write to csv files
csv_path = "/home/haitongc/Shoes-ACOSI-models/EECS595Project/data/score_comp/"
for a in annotation:
    csv_file = csv_path + task + "_" + dataset + "_" + a + ".csv"
    # csv_file = task + "_" + dataset + "_" + a + ".csv"
    with open(csv_file, 'w', newline='') as c:
        writer = csv.writer(c)
        header = ["model"] + field
        writer.writerow(header)

        # list to be written to csv, model name fllowed by scores
        l = []
        for key in model_scores[a]:
            # add model name to list
            l.append(key)
            scores = model_scores[a][key]
            for score in scores:
                # add all scores to list
                l.append(scores[score])
            writer.writerow(l)
            l = []

