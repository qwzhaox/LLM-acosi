from itertools import chain
from ast import literal_eval

str1 = "['i can't ', 'right?']"
# list1 = eval(str1)

# print(list1)

list2 = literal_eval(str1)

print(list2)
