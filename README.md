# LLM-acosi

## Setup
Please use Python 3.10

Setup environment
```
python3 -m venv env
pip install -r requirements.txt
```

Download "punkt" from nltk
```
python3
>>> import nltk
>>> nltk.download("punkt")
```

## File Structure
### data/
acos/ - contains acos dataset files <br>
acosi/ - contains acosi dataset files <br>
eval_output/meta-llama/ - evaluation scores of Llama models in .json format <br>
model_output/meta-llama/ - Llama model output in .pkl format <br>
score_comp/ - comparison tables of evaluation scores of all Llama models in .csv format <br>
t5_output/ - output of t5 model

### scripts/
**bash**/ - .sh scripts to run .py files <br>
acos_extend.py - script to run acos_extend task <br>
acos_extract.py - script to run acos_extract task <br>
acosi_extract.py - script to run acosi_extract task <br>
get_scores_csv.py - script to get the comparison tables and output to .csv files from model output in .json format <br>
pipeline.py - script to run task using Hugging Face pipeline <br>
pkl_to_json.py - script to convert llm output in .pkl files to .json format <br>
utils.py - helper functions to be used in multiple scripts
