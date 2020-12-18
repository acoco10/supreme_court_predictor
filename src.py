#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd 
import json
import os
import numpy as np
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import punkt
import nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
import string
from nltk.probability import FreqDist
import seaborn as sns



def get_lawyers (case, justices):
    lawyers = []
    for x in case: 
        if x[0] in justices or x[0] in lawyers:
            continue
        else:
            lawyers.append(x[0])
    return lawyers


# In[ ]:


def get_petitioner_words (case, justices, caseldict, sct):
    petitioner_words = []
    words = sct[case]
    previous_speaker = words[0][0]
    for speaker in words:
        if speaker[0] == caseldict[case][0]:
            petitioner_words.append(speaker[1])
        if speaker[0] in justices and caseldict[case][0] == previous_speaker:
            petitioner_words.append(speaker[1])  
        previous_speaker = speaker[0]
    return petitioner_words


# In[ ]:


def get_justice_words (case, justices, caseldict, sct):
    justice_words = []
    words = sct[case]
    previous_speaker = words[0][0]
    for speaker in words:
        if speaker[0] in justices and caseldict[case][0] == previous_speaker:
            justice_words.append(speaker[1])  
        previous_speaker = speaker[0]
    return justice_words


# In[ ]:


def remove_non_ascii_chars(title):
    return "".join([unidecode(char).rstrip('()').rstrip(' ') for char in title])      


# In[ ]:

sw_list = stopwords.words('english')
sw_list += list(string.punctuation)
sw_list += ["''", '""', '...', '``', '’', '“', '’', '”', '‘', '‘', '©',"'"
            'said', 'one', 'com', 'http', '-', '–', '—', 'co', 'wa', 'ha', '1', 'amp','court', 'would', 'case', 'say', 'think', 'state', 'well', 'make',
       'right', 'question', 'mr', 'go', 'could', 'statute', 'yes',
       'honor', 'fact', 'justice', 'law', 'time']
sw_set = set(sw_list)

def process_article(article):
    tokens = nltk.word_tokenize(article)
    stopwords_removed = [token.lower() for token in tokens if token.lower() not in sw_set]
    return stopwords_removed 







# In[ ]:

sw_list2 = sw_list = stopwords.words('english')
sw_list2 += list(string.punctuation)
sw_list2 += ["''", '""', '...', '``', '’', '“', '’', '”', '‘', '‘', '©',"'"
            'said', 'one', 'com', 'http', '-', '–', '—', 'co', 'wa', 'ha', '1', 'amp','court', 'would', 'case', 'say', 'think', 'state', 'well', 'make',
       'right', 'question', 'mr', 'go', 'could', 'statute', 'yes',
       'honor', 'fact', 'justice', 'law', 'time' 'may',
 'whether',
 'take',
 'get',
 'act',
 'know',
 'point',
 'issue',
 'first',
 'rule',
 'give',
 'government',
 'federal',
 'two',
 'congress',
 'judge',
 'appeal',
 'district',
 'mean',
 'use'] 

sw_set2 = set(sw_list2)


# In[ ]:

def process_article2(article):
    tokens = nltk.word_tokenize(article)
    stopwords_removed = [token.lower() for token in tokens if token.lower() not in sw_set2]
    return stopwords_removed 




# In[ ]:





# In[ ]:




