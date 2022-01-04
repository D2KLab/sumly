### Libraries ###
import argparse

from Bio import Entrez
import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import os

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
from collections import Counter
from scipy.stats import entropy
from scipy.spatial import distance
import statistics as sc

from string import punctuation
import en_core_web_md
nlp = en_core_web_md.load()

import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import matplotlib.pyplot as plt


### Information Retrieval ###
def search(query, max_articles, start_date, end_date, email):
  Entrez.email = email

  # Get the PubMed ID of articles according to the query
  handle = Entrez.esearch(db='pubmed', sort='relevance', retmax=max_articles, retmode='xml', term=query, mindate=start_date, maxdate=end_date)
  results = Entrez.read(handle)

  return results

def fetch_details(id, email):
  Entrez.email = email

  # Get the PubMed details of the article
  handle = Entrez.efetch(db='pubmed', retmode='xml', id=id)
  results = Entrez.read(handle)

  return results

def get_text(id):

  # Check if article can be accessed
  url = f'http://www.ncbi.nlm.nih.gov/pmc/articles/pmid/{id}'
  try:
    response = requests.get(url)
  except:
    return

  soup = BeautifulSoup(response.content, features='html.parser')
  
  # Check if the main contents can be scrapped
  div = soup.findAll('p', {'id': re.compile('.*p.*', re.IGNORECASE)})
  if len(div) == 0:
    return
  
  # Scrape the main contents of the article
  text = ''
  for i, tag in enumerate(div):
    sentence = ' '.join(string.strip() for string in tag.strings)
    if i == 0:
      text += sentence
    else:
      text += ' ' + sentence
  
  return text


### Metrics ###       
def jaccard(org, summ):

  # List the unique words in a document
  words_doc1 = set(str(org))
  words_doc2 = set(str(summ))

  # Find the intersection of words list of doc1 & doc2
  intersection = words_doc1.intersection(words_doc2)
  
  # Find the union of words list of doc1 & doc2
  union = words_doc1.union(words_doc2)
  
  # Using length of intersection set divided by length of union set
  return float(len(intersection)) / len(union)

def cosine(org, summ):
  X_list = word_tokenize(str(org))
  Y_list = word_tokenize(str(summ))
  sw = stopwords.words('english')
  
  l1 =[]
  l2 =[]
  X_set = {w for w in X_list if not w in sw}
  Y_set = {w for w in Y_list if not w in sw}

  rvector = X_set.union(Y_set)
  for w in rvector:
    if w in X_set: l1.append(1)
    else: l1.append(0)
    if w in Y_set: l2.append(1)
    else: l2.append(0)
  
  c = 0
  for i in range(len(rvector)):
    c += l1[i]*l2[i]
  cosine = c / float((sum(l1)*sum(l2))**0.5)
  return cosine

def kld(org, summ): 
  dist_original = Counter(org.lower().split())
  dist_summary = Counter(summ.lower().split())
  p = list(dist_original.values())
  q = list(dist_summary.values())
  a = min(len(p), len(q))
  return entropy(p[0:a], q[0:a])
 
def jsd(org, summ):
  dist_original = Counter(org.lower().split())
  dist_summary = Counter(summ.lower().split())
  p = list(dist_original.values())
  q = list(dist_summary.values())
  a = min(len(p), len(q))
  return distance.jensenshannon(p[0:a], q[0:a])


### Statistical Summarization ### 
def statistical(text):
  keyword = []
  pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
  doc = nlp(text.lower())
  for token in doc:
    if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
      continue
    if(token.pos_ in pos_tag):
      keyword.append(token.text)

  freq_word = Counter(keyword)
  max_freq = Counter(keyword).most_common(1)[0][1]
  for w in freq_word:
      freq_word[w] = (freq_word[w] / max_freq)
      
  sent_strength = {}
  for sent in doc.sents:
    for word in sent:
      if word.text in freq_word.keys():
        if sent in sent_strength.keys():
          sent_strength[sent] += freq_word[word.text]
        else:
          sent_strength[sent] = freq_word[word.text] 
  
  summary = []
  sorted_x = sorted(sent_strength.items(), key=lambda kv: kv[1], reverse=True)
  limit = len(sorted_x) / 15
  counter = 0

  for i in range(len(sorted_x)):
    summary.append(str(sorted_x[i][0]).capitalize())
    counter += 1
    if counter >= limit:
      break
          
  return ' '.join(summary)


### Transformer Summarization ###  
def find_attentions(sentences):
  input_text = []
  for sent in sentences:
    input_text.append(tokenizer.encode_plus(sent, add_special_tokens=True, return_tensors='pt', truncation=True, max_length=512))

  tensor_text = pad_sequence([item['input_ids'].squeeze(0) for item in input_text], batch_first=True)
  tensor_mask  = pad_sequence([item['attention_mask'].squeeze(0) for item in input_text], batch_first=True)
  tensor_ids = pad_sequence([item['token_type_ids'].squeeze(0) for item in input_text], batch_first=True) 

  tensor_text, token_type, attention_mask = tensor_text.cuda(), tensor_ids.cuda(), tensor_mask.cuda()
    
  with torch.no_grad():
    attentions = model(tensor_text, token_type_ids = token_type, attention_mask = attention_mask)[-1]
    final_score = [np.max(attentions[sen][:].cpu().numpy()) for sen in range(attentions.shape[0])]
    final_score = [0 if i<0 else i for i in final_score]
  
  return final_score

def transformer(text):
  doc = nlp(text)
  sentences = [sent.string.strip() for sent in doc.sents]  

  attentions = find_attentions(sentences)
  lengths = [len(sent) for sent in sentences]

  scores = [[a] * leng for a, leng in zip(attentions, lengths)]
  scores = sum(scores, [])
  
  words = [word_tokenize(sent) for sent in sentences]
  extraction = [word for (word, attention) in zip(words, scores) if float(attention) > sc.mean(attentions)]
  
  summary = ''
  for i, sent in enumerate(extraction):
    if i == 0:
      summary += ' '.join(sent)[:-2] + sent[-1]
    else:
      summary += ' ' + ' '.join(sent)[:-2] + sent[-1]

  return summary


### Main Function ###
def main():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    
    retrieve = subparser.add_parser('retrieve')
    summarize = subparser.add_parser('summarize')
    
    retrieve.add_argument('--query', type=str, required=True, help='Query')
    retrieve.add_argument('--max_articles', type=str, required=True, help='Max number of articles')
    retrieve.add_argument('--start_date', type=str, required=True, help='Start date (yyyy/mm/dd)')
    retrieve.add_argument('--end_date', type=str, required=True, help='End date (yyyy/mm/dd)')
    retrieve.add_argument('--output_path', type=str, required=True, help='Output directory')
    retrieve.add_argument('--email', type=str, required=True, help='Email')
    
    summarize.add_argument('--input_path', type=str, required=True, help='Input directory')
    summarize.add_argument('--output_path', type=str, required=True, help='Output directory')
    summarize.add_argument('--summ_type', type=str, required=True, help='Summarization type')
    
    args = parser.parse_args()
    
    # Retrieve and save the PubMed articles
    if args.command == 'retrieve':
    
        # Initialize the parameters
        query = args.query
        max_articles = args.max_articles
        start_date = args.start_date
        end_date = args.end_date
        output_path = args.output_path
        email = args.email
        
        # Check if output path is a file directory
        if not os.path.isdir(output_path):
            raise ValueError('File directory does not exist.')
        
        # Get the PubMed IDs of articles based on the query
        results = search(query, max_articles, start_date, end_date, email)
        id_list = results['IdList']

        # Get the title of the PubMed articles
        titles = {}
        for id in id_list:
          paper = fetch_details(id, email)
          titles[id] = paper['PubmedArticle'][0]['MedlineCitation']['Article']['ArticleTitle']
         
        # Scrape the text of the PubMed articles
        articles = []
        success = []

        for id in tqdm(id_list):
          text = get_text(id)
          if text is not None:
            articles.append(text)
            success.append(id)

        print(f'\nNumber of articles scrapped: {len(articles)}\n')
        for id in success:
            print(f'[{id}]: {titles[id]}')
        
        # Save each PubMed article as a text file
        for i, article in enumerate(articles):
          with open(output_path + f'/article_{success[i]}.txt', 'w') as text_file:
              text_file.write(article)
    
    # Summarize the text extractively
    elif args.command == 'summarize':

        # Initialize the parameters
        input_path = args.input_path
        output_path = args.output_path
        summ_type = args.summ_type
        
        # Check if input path is a filename
        if os.path.isfile(input_path):
            
            # Check if output path is a file directory
            if os.path.isdir(output_path):
                input_type = 'single'
            
                with open(input_path, 'r') as text_file:
                    text = text_file.read()               
            else:
                raise ValueError('Output path must be a file directory.')
        
        # Check if input path is a file directory
        elif os.path.isdir(input_path):
        
            # Check if output path is a file directory
            if os.path.isdir(output_path):
                input_type = 'multiple'
            
                # Check if there are files in the file directory
                files = os.listdir(input_path)
                if files is None:
                    raise ValueError('No files found.')
                
                texts = []
                for file in files:
                  with open(input_path + '/' + file, 'r') as text_file:
                      texts.append(text_file.read())                
            else:
                raise ValueError('Output path must be a file directory.')
        
        else:
            raise ValueError('Input path must be a filename or file directory and output path must be a file directory.')     
        
        # Summarize using the statistical method
        if summ_type == 'statistical':
            print('\nStatistical Based Summarization')
            
            if input_type == 'single':
            
                # Summarize the text
                summary = statistical(text)
                
            elif input_type == 'multiple':
            
                # Summarize the texts
                summaries = []
                for i, text in enumerate(tqdm(texts)):
                    try:
                        summary = statistical(text)
                        summaries.append(summary)
                    except:
                        print(f'Error with input: {files[i]}')         
        
        # Summarize using the transformer method
        elif summ_type == 'transformer':
            print('\nTransformer Based Summarization')
            
            # Check whether there is a GPU
            cuda_flag = torch.cuda.is_available()
            if cuda_flag:
              device = torch.cuda.current_device()
              device_name = torch.cuda.get_device_name(device)
              
              print(f'\nCurrent device: {device_name}\n')
            else:
              raise ValueError('GPU not found.')           
            
            # Initialize the tokenizer and model
            global tokenizer
            global model
            tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
            model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)            
            
            if input_type == 'single':
            
                # Summarize the text
                summary = transformer(text)
                
            elif input_type == 'multiple':
            
                # Summarize the texts
                summaries = []
                for i, text in enumerate(tqdm(texts)):
                    try:
                        summary = transformer(text)
                        summaries.append(summary)
                    except:
                        print(f'Error with input: {files[i]}')                 
        
        # Calculate the individual metric scores
        if input_type == 'single':
            print()
            print(f'KLD Frequency: {round(kld(text, summary), 3)}')
            print(f'JSD Frequency: {round(jsd(text, summary), 3)}')
            # print(f'Jaccard Similarity: {round(jaccard(text, summary), 3)}')
            # print(f'Cosine Similarity: {round(cosine(text, summary), 3)}')
               
            # Save the summary as a text file
            with open(output_path + f"/summary_{input_path.split('/')[-1].split('.')[0]}.txt", 'w') as text_file:
                text_file.write(summary)             
        
        # Calculate the average metric scores
        elif input_type == 'multiple':
            kld_freq = []
            jsd_freq = []
            # jaccard_sim = []
            # cosine_sim = []
            for org, summ in zip(texts, summaries):
                kld_freq.append(kld(org, summ))
                jsd_freq.append(jsd(org, summ))
                # jaccard_sim.append(jaccard(org, summ))
                # cosine_sim.append(cosine(org, summ))

            print()
            print(f'Average KLD Frequency: {round(sc.mean(kld_freq), 3)}')
            print(f'Average JSD Frequency: {round(sc.mean(jsd_freq), 3)}')
            # print(f'Average Jaccard Similarity: {round(sc.mean(jaccard_sim), 3)}')
            # print(f'Average Cosine Similarity: {round(sc.mean(cosine_sim), 3)}')
            
            plt.style.use('ggplot')
            plt.figure(figsize=(10, 8))
            plt.plot(kld_freq, 'b', label='KL Divergance')
            plt.plot(jsd_freq, 'r', label='JS Divergance')
            # plt.plot(jaccard_sim, 'y', label='Jaccard Similarity')
            # plt.plot(cosine_sim, 'g', label='Cosine Similarity')
            plt.xlabel('Dataset')
            plt.ylabel('Closeness to Original Document')
            plt.legend(loc='upper right')
            if summ_type == 'statistical':
                plt.title('Statistical Based')
                plt.savefig('chart.png')
            elif summ_type == 'transformer':
                plt.title('Transformer Based')
                plt.savefig('chart.png')

            # Save each summary as a text file
            for i, summary in enumerate(summaries):
              with open(output_path + f"/summary_{files[i].split('.')[0]}.txt", 'w') as text_file:
                  text_file.write(summary)    

main()