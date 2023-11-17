import json
import numpy
import random
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader

class T5_reader(IterableDataset):
    def __init__(self, file):
        super(T5_reader).__init__()
        self.file = file
        print('Load T5_reader succsessfully')

    def __iter__(self):

        with open(self.file, 'r') as f:
            data = []
            for line in f:
                if line != '':
                    tmp_dic = {}
                    review, label = line.strip().split('####')
                    tmp_dic['review'] = review.strip()
                    tmp_dic['labels'] = eval(label.strip())
                    data.append(tmp_dic)
        for x in data:
            # print(x['review'], '<labels>' + '<labels>'.join([' '.join (i) for i in x['labels']]))
            for i in range(len(x['labels'])):
                for idx, a_word in enumerate(x['labels'][i]):
                    if idx == 0:
                        x['labels'][i][idx] = '<A>' + x['labels'][i][idx]
                    if idx == 1:
                        x['labels'][i][idx] = '<C>' + x['labels'][i][idx]
                    if idx == 2:
                        x['labels'][i][idx] = '<S>' + x['labels'][i][idx]
                    if idx == 3:
                        x['labels'][i][idx] = '<O>' + x['labels'][i][idx]
                    if idx == 4:
                        x['labels'][i][idx] = '<I>' + x['labels'][i][idx]

            yield x['review'], '<labels>'.join([' '.join (i) for i in x['labels']])
            # yield x['review'], '<labels>' + '</s>'.join([' '.join (i) for i in x['labels']])

class Ryan_reader(IterableDataset):
    def __init__(self, file):
        super(Ryan_reader).__init__()
        self.file = file
        print('Load Ryan_reader succsessfully')

    def __iter__(self):

        with open(self.file, 'r') as f:
            data = []
            for line in f:
                if line != '':
                    tmp_dic = {}
                    review, label = line.strip().split('####')
                    tmp_dic['review'] = review.strip()
                    data.append(tmp_dic)
        for x in data:
            # print(x['review'], '<labels>' + '<labels>'.join([' '.join (i) for i in x['labels']]))
            for i in range(len(x['labels'])):
                for idx, a_word in enumerate(x['labels'][i]):
                    if idx == 0:
                        x['labels'][i][idx] = '<A>' + x['labels'][i][idx]
                    if idx == 1:
                        x['labels'][i][idx] = '<C>' + x['labels'][i][idx]
                    if idx == 2:
                        x['labels'][i][idx] = '<S>' + x['labels'][i][idx]
                    if idx == 3:
                        x['labels'][i][idx] = '<O>' + x['labels'][i][idx]
                    if idx == 4:
                        x['labels'][i][idx] = '<I>' + x['labels'][i][idx]

            yield x['review'], '<labels>'.join([' '.join (i) for i in x['labels']])