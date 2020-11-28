#!/usr/bin/env python3
import torch
import transformers
import sys
import numpy as np
import jsonlines as js
from typing import Optional
from typing import Sequence
import getopt
import argparse
import pprint
import json
import requests
import sys
import csv
import os
import re
import numpy as np
import pandas as pd
import statistics as sc
from scipy.stats import entropy
from scipy.spatial import distance
from collections import Counter
from collections import OrderedDict
from collections import Counter
from string import punctuation
import textstat
from textstat.textstat import textstat
import en_core_web_lg
nlp = en_core_web_lg.load()





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
    #print('Size of maximum string is ',padsize)
    #print('Total number of lists ',listsize)
   # print('Longest Sentence is at index ',pos_longestsentence)
    return (padsize,listsize,pos_longestsentence)



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



def extract_bert_summary(summary):
  attentions = find_attentions(summary)
  lengths=[len(lst) for lst in summary]
  score=[[a]*leng for a,leng in zip(attentions,lengths)]
  score = sum(score,[])
  words = [word  for sentence in summary for word in sentence]
  extraction = [word for (word,attention) in zip(words,score) if float(attention) > sc.mean(attentions)]
  #print(extraction)
  return ' '.join(extraction)

def summarize(text):
    keyword = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB', 'NUM', 'NN' ]
    #doc=nlp(text)type
    # doc = nlp(text.lower() if isinstance(text, str) else text)
    doc = nlp(str(text).lower())

    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            keyword.append(token.text)

    freq_word = Counter(keyword)
    most_common = freq_word.most_common(1)
    max_freq = None
    if most_common and len(most_common[0])>1:
        max_freq = most_common[0][1]
    # print(freq_word.most_common(1))
    # max_freq = Counter(keyword).most_common(1)[0][1]

    for w in freq_word:
        freq_word[w] = (freq_word[w]/max_freq)

    sent_strength={}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent]+=freq_word[word.text]
                else:
                    sent_strength[sent]=freq_word[word.text]
    summary = []

    sorted_x = sorted(sent_strength.items(), key=lambda kv: kv[1], reverse=True)
    limit=len(sorted_x)/15
    counter = 0
    for i in range(len(sorted_x)):
        summary.append(str(sorted_x[i][0]).capitalize())

        counter += 1
        if(counter >= limit):
            break

    return ' '.join(summary)


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
    parser.add_argument('transformation', type=str, help='transform type')

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

    if(args.transformation == "statistical"):
      print("Statistical-based model ...")
      summary=[summarize(note) for note in clinical_notes]
    elif( args.transformation == "transformer"):
      print("Transformer-based model ...")
      summary=[extract_bert_summary(note) for note in clinical_notes]
    else:
      print("Error happened, please check input parameter")



    #clinical_notes = Book.append(summary)



    print(summary )


    save(filename=args.outputfile, summary=summary)




main()





