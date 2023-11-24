from itertools import chain

hi = [{"hi": "hf"}, {"hello": "helflo"}]

print(list(chain(*hi)))
