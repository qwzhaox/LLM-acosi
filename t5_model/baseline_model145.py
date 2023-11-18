import argparse
import random
import statistics
import torch
from transformers import AutoModelForSeq2SeqLM as Model
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoConfig, AutoModelForSeq2SeqLM
import torch.nn as nn
from rouge_score import rouge_scorer
import copy
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True).score
from torch.utils.data import Dataset, IterableDataset, DataLoader
from data_reader import T5_reader
import os
import numpy as np
from tqdm import tqdm

tokenizer = T5Tokenizer.from_pretrained("t5-base")


def calculate(list_a, list_b):
    rouge_score = []
    for i in range(len(list_a)):
        rouge_score.append(scorer(list_a[i], list_b[i])['rougeL'][-1])
    return rouge_score


def naive_train_T5(args):
    print('those are the parameters used for finetuning T5 %s' % str(args))
    tokenizer.add_tokens(['<labels>', '<A>', '<C>', '<O>', '<S>', '<I>'])
    dataset = T5_reader(args.train_data)
    dataloader = DataLoader(dataset, batch_size = args.batch_size)
    t_dataset = T5_reader(args.test_data)
    t_dataloader = DataLoader(t_dataset, batch_size = args.batch_size)
    model = T5ForConditionalGeneration.from_pretrained(args.t5_version)
    model.resize_token_embeddings(len(tokenizer))
    model.cuda()
    epoch = args.epochs
    lr = args.learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100)
    ep = 0
    rouge_iter = 0
    checker = 0
    while ep <=epoch:
        losses = []
        for batch_idx, (inp, oup) in tqdm(enumerate(dataloader)):
            model.train()
            inp_encoding = tokenizer(inp,
            padding = 'longest',
            max_length=args.max_input_length,
            truncation=True,
            return_tensors = 'pt',
            )
            inp_ids, attention_mask = inp_encoding.input_ids.cuda(), inp_encoding.attention_mask.cuda()
            target_encoding = tokenizer(oup,
            padding="longest",
            max_length=args.max_target_length,
            truncation=True,
            return_tensors="pt",
            )
            output_ids, output_mask = target_encoding.input_ids.cuda(), target_encoding.attention_mask.cuda()
            output_ids[output_ids == tokenizer.pad_token_id] = -100
            pred = model(inp_ids, attention_mask, decoder_attention_mask=output_mask, labels=output_ids,
            output_hidden_states=True)
            loss = pred.loss
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            ep += 1
            checker += 1
            if checker % args.check_every == 0:
                print('Current step is %d loss is %.5f' % (checker, np.mean(losses)))
                model.eval()
                rouge_scorer = []
                pred_sums = []
                for _, (inp, oup) in tqdm(enumerate(t_dataloader)):
                    inp = tokenizer(inp,
                    padding = 'longest',
                    max_length=args.max_input_length,
                    truncation=True,
                    return_tensors = 'pt',
                    )
                    inp_ids, attention_mask = inp.input_ids.cuda(), inp.attention_mask.cuda()
                    preds = model.generate(input_ids = inp_ids,
                    attention_mask = attention_mask,
                    max_length = 64,
                    num_beams = 1,
                    no_repeat_ngram_size=3,
                    early_stopping = True
                    )
                    for pred in preds:
                        pred_sums.append(tokenizer.decode(pred, skip_special_tokens=True))
                bt_scores = calculate(list(oup), pred_sums)
                rouge_scorer += bt_scores
                c_rouge = np.mean(rouge_scorer)
                print('The first label is %s' % str(oup))
                print('the first pred is %s' % str(pred_sums))
                """
                if c_rouge > rouge_iter:
                    os.makedirs(
                    args.model_path + '/' + str(args.epochs), exist_ok=True
                    )
                    print('saving a model')
                    rouge_iter = c_rouge
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': np.mean(losses)
                    },
                    args.model_path + '/' + str(args.epochs) + '/' + 'Best_step%dRouge%.3f'%(checker, c_rouge)
                    )
                """
            if checker == args.epochs:
                break



def predict_rest(model_path, args):

    tokenizer.add_tokens(['<labels>', '<A>', '<C>', '<O>', '<S>', '<I>'])
    model = Model.from_pretrained(args.t5_version, return_dict=True)
    model.resize_token_embeddings(len(tokenizer))
    # model.load_state_dict(torch.load(model_path))
    # best_point = torch.load(model_path)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    model.cuda()
    

    print('current mode  is %s' % args.mode)
    rest_data  = args.rest_data
    with open (rest_data, 'r') as f:
        for line in f:
            inp_encoding = tokenizer(line,
                                    return_tensors = 'pt',
                                    )
            inp_ids, attention_mask = inp_encoding.input_ids.cuda(), inp_encoding.attention_mask.cuda()
            
            pred = model.generate(
                input_ids = inp_ids,
                attention_mask = attention_mask,
                max_length = 256,
                num_beams = 3,
                no_repeat_ngram_size=5,
                early_stopping = True
            )
            print(tokenizer.decode(pred[0], skip_special_tokens=True))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default='train', type=str)
    parser.add_argument('-t5_version', default="google/t5-v1_1-base", type=str)
    parser.add_argument('-epochs', default=30000, type=int)
    parser.add_argument('-learning_rate', default=1e-4, type=float)
    parser.add_argument('-batch_size', default=8, type=int)
    parser.add_argument('-train_data', default='/home/jade/ACOSI/data/shoes_966.txt', type=str)
    parser.add_argument('-test_data', default='/home/jade/ACOSI/data/shoes_test.txt', type=str)
    parser.add_argument('-model_path', default='Naive', type=str)
    parser.add_argument('-shuffle', default=True, type=bool)
    parser.add_argument('-max_input_length', default=128, type=int)
    parser.add_argument('-max_target_length', default=64, type=int)
    parser.add_argument('-check_every', default=200, type=int)
    parser.add_argument('-rest_data', default='/home/jade/ACOSI/data/ryan.txt', type=str)


    args = parser.parse_args()
    naive_train_T5(args) 
    
    """
    if args.mode == 'train':
        naive_train_T5(args)
    elif args.mode == 'predict':
        predict_rest(
            '/home/jade/ACOSI/code/model/Naive/30000/Best_step29000Rouge0.430',
            args
        )
    """
    print('This is a placeholder')
