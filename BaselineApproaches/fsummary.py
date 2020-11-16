
#!/usr/bin/env python3
import numpy
import jsonlines as js
import getopt
import argparse
import pprint
from typing import Optional
from typing import Sequence
import ndjson
import json
import requests
import transformers
import sys
import torch
import numpy
import statistics
import scipy
import collections
import os
import csv
import tensorflow
import pandas as pd
import numpy as np
import statistics as sc
import spacy


from scipy.stats import entropy
from scipy.spatial import distance
from collections import Counter
from collections import OrderedDict
from collections import Counter



from string import punctuation

import en_core_web_lg
nlp = en_core_web_lg.load()




def read_csv(csvfile):
    print('read_csv(): type(csvfile)) = {}'.format(csvfile))

    #spacy.load('en_core_web_lg')
    foo_df = pd.read_csv(csvfile)

    return foo_df


def kld(summ, org):
        dist_original=Counter(org.lower().split())
        dist_summary=Counter(summ.lower().split())
        q = list(dist_original.values())
        p=list(dist_summary.values())
        a=min(len(p),len(q))
        return entropy(p[0:a],qk=q[0:a])
def jsd(summ,org):
        dist_original=Counter(org.lower().split())
        dist_summary=Counter(summ.lower().split())
        p = list(dist_original.values())
        q = list(dist_summary.values())
        a=min(len(p),len(q))
        return distance.jensenshannon(p[0:a],q[0:a])


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

#
def save(filename, summary):
    outF = open(filename, "w")
    for line in summary:

        outF.write(line)
        #outF.write("\n")
        outF.write(" ")
    outF.close()

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

#Book.size



    #clinical_notes = [[' '.join( sum(Book[i],[0]))] for i in range(100)]
    ##clinical_notes = [Book[i][0 ]for i in range(len(foo_df))]

    summary=[summarize(note) for note in clinical_notes]


    #clinical_notes = Book.append(summary)



    print(summary )

    save(filename=args.outputfile, summary=summary)


    #
    # dist_original=Counter(clinical_notes[0].lower().split())
    # dis_summary=Counter(summary[0].lower().split())
    #
    # print(list(dist_original.values()))
    # print(list(dis_summary.values()))
    # kld_freq=[kld(clinical_notes[i],summary[i]) for i in range(10)]
    # print(kld_freq)
    # jsd_freq=[jsd(summ,org) for org,summ in zip(clinical_notes,summary)]
    # print(jsd_freq)
    # print(sc.mean(kld_freq))
    # print(sc.mean(jsd_freq))

main()
