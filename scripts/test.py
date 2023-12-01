from itertools import chain
from ast import literal_eval
from utils import is_special_token, needs_closing_bracket, SSEP, END, ACOSI

prediction = "[A [SSEP [END] [ [A ["

insert_loc = []
delete_loc = len(prediction)

insert_loc = []
delete_loc = len(prediction)

for i, char in enumerate(prediction):
    if char == "[":
        start = i + 1
        SSEP_end = start + len(SSEP)
        END_end = start + len(END)
        ACOSI_end = start + 1
        if is_special_token(start, SSEP_end, prediction, SSEP):
            if needs_closing_bracket(SSEP_end, prediction):
                insert_loc.append(SSEP_end)
        elif is_special_token(start, END_end, prediction, END):
            if needs_closing_bracket(END_end, prediction):
                insert_loc.append(END_end)
        elif is_special_token(start, ACOSI_end, prediction, ACOSI):
            if needs_closing_bracket(ACOSI_end, prediction):
                insert_loc.append(ACOSI_end)
        elif i + 1 == len(prediction):
            delete_loc = i

prediction = prediction[:delete_loc]
for i in reversed(insert_loc):
    prediction = prediction[:i] + "]" + prediction[i:]

prediction = prediction.strip()

print(prediction)
