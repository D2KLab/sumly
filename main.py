#!/usr/bin/env python3
import torch

import sys
import numpy as np
import jsonlines as js
import getopt
import argparse
import pprint
import json
import requests
import sys
import csv
import os
import re
import pandas as pd

def read_csv(csvfile):
    print('read_csv(): type(csvfile))={}'.format(csvfile))
    foo_df = pd.read_csv(csvfile)
    return foo_df


cuda_flag = torch.cuda.is_available()
if cuda_flag:
    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
print('current device = '+device_name)



# book = []
# with js.open("notes_labeled_dev.ndjson") as reader:
#   num = 0
#   for summary in reader:
#     if num<100:
#         book.append(summary[2])
#         num = num + 1

# clinical_notes = [" ".join( sum(book[i],[])) for i in range(100)]
# Book=[]
# with js.open('notes.ndjson','r') as reader:
#     num = 0
#     for summary in reader:
#       if num<10:
#         Book.append( summary)
#         num = num + 1


#clinical_notes = [Book[i][2] for i in range(10)]
#print(clinical_notes)
global model
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


#print(clinical_notes)
def find_stats(note):
    padsize=0
    listsize=0
    pos,pos_longestsentence=0,0
    for lst in note:
        listsize+=1
        padsize=max(padsize,len(lst))
        if len(lst)==padsize:
            pos_longestsentence=pos
            pos+=1
    print('Size of maximum string is ',padsize)
    print('Total number of lists ',listsize)
    print('Longest Sentence is at index ',pos_longestsentence)
    return (padsize,listsize,pos_longestsentence)

import numpy as np

from torch.nn.utils.rnn import pad_sequence
if cuda_flag:
    model = model.cuda()
model.eval();

def find_attentions(summary):

    input_text = []
    for lst in summary:
        input_text.append(tokenizer.encode_plus(lst, add_special_tokens=True, return_tensors='pt'))  # ecnode
    # do padding to make all sentences equal for BERT
    tensor_text = pad_sequence([item['input_ids'].squeeze(0) for item in input_text], batch_first=True)
    tensor_mask = pad_sequence([item['attention_mask'].squeeze(0) for item in input_text], batch_first=True)
    tensor_ids = pad_sequence([item['token_type_ids'].squeeze(0) for item in input_text], batch_first=True)

    if cuda_flag:
        tensor_text, token_type, attention_mask = tensor_text.cuda(), tensor_ids.cuda(), tensor_mask.cuda()

    with torch.no_grad():
        # loss,logits,attentions = model(tensor_text, token_type_ids = token_type, attention_mask = attention_mask)
        attentions = model(tensor_text, token_type_ids=token_type, attention_mask=attention_mask)[-1]
        # final_score=attentions[11].mean(axis=0).mean(axis=0).cpu().numpy()
        final_score = [np.max(attentions[sen][:].cpu().numpy()) for sen in range(attentions.shape[0])]
        final_score = [0 if i < 0 else i for i in final_score]
        score_norm = [i / sum(final_score) for i in final_score]
    return final_score

def save(filename, summary):
    if filename.endswith(".txt"):
        outF = open(filename, "w")
        for line in summary:
            outF.write(str(line))
            #outF.write("\n")
            outF.write(" ")
        outF.close()
    elif filename.endswith(".jsonl"):
        out_file = open(filename, "w")
        json.dump({"text": "".join(summary).strip()}, out_file, indent = 4, sort_keys = False)
        out_file.close()
def main():
    Book=[]
    parser = argparse.ArgumentParser(description='Make barchart from csv.')
    parser.add_argument('-d', '--debug', help='Debugging output', action='store_true')
    parser.add_argument('csvfile', type=argparse.FileType('r'), help='Input csv file')
    parser.add_argument('outputfile', type=str, help='Output csv file')
    args = parser.parse_args()
      #print( format(args.csvfile))
   # print('main(): type(args.csvfile)) = {}'.format(args.csvfile))
    #print('')

    #csv='ADMISSIONS.csv'
    #foo_df = pd.read_csv(csv)
    foo_df = pd.read_csv(args.csvfile)
    Book=foo_df.values
    print(f'Book size: {Book.size}')
    clinical_notes = []


    clinical_notes = [Book[i][0] for i in range(Book.size) if Book[i]]
    (pad_size, _, _) = find_stats(clinical_notes[0])
    print(pad_size)
#Book.size or insert line number



    #clinical_notes = [[' '.join( sum(Book[i],[0]))] for i in range(100)]
    ##clinical_notes = [Book[i][0 ]for i in range(len(foo_df))]

    summary=[find_attentions(note) for note in clinical_notes]


    #clinical_notes = Book.append(summary)



    print(summary )

    save(filename=args.outputfile, summary=summary)




main()