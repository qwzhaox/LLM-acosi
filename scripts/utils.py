from pathlib import Path
from ast import eval


def get_file_path(file_name):
    # Search in the current directory and all subdirectories
    for path in Path(".").rglob(file_name):
        # Return the first match
        return path
    # Return None if no match is found
    return None


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