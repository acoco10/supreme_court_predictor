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
from unidecode import unidecode
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
def get_justices (case, justices):
    justicesls = []
    for x in case: 
        if x[0] in justices:
            justicesls.append(x[0])
    return list(set([x[0] for x in case if x[0] in justices]))


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

def remove_non_ascii_chars_t(title):
    return "".join([unidecode(char) for char in title])      
# In[ ]:

sw_list = stopwords.words('english')
sw_list += list(string.punctuation)
sw_list += ["''", '""', '...', '``', '’', '“', '’', '”', '‘', '‘', '©',"'"
            'said', 'one', 'com', 'http', '-', '–', '—', 'co', 'wa', 'ha', '1', 'amp','court', 'would', 'case', 'say', 'think', 'state', 'well', 'make','right', 'question', 'mr', 'go', 'could', 'statute', 'yes','honor', 'fact', 'justice', 'law', 'time', 'may','whether', 'take', 'get', 'act', 'know', 'point', 'issue', 'first', 'rule', 'give', 'government', 'federal', 'two', 'congress', 'judge','appeal', 'district','mean','use' 'may', 'it', 'please', 'the', 'court', 'justice', 'thank', 'you', 'mrs'] 
sw_set = set(sw_list)

def process_article(article):
    tokens = nltk.word_tokenize(article)
    stopwords_removed = [token.lower() for token in tokens if token.lower() not in sw_set]
    return stopwords_removed 
lemmatizer = WordNetLemmatizer()
def lemm_text(words):
    lem = []
    for word in words:
        lem.append(lemmatizer.lemmatize(word))
    return lem  

class W2vVectorizer(object):
    
    def __init__(self, w2v):
        # Takes in a dictionary of words and vectors as input
        self.w2v = w2v
        if len(w2v) == 0:
            self.dimensions = 0
        else:
            self.dimensions = len(w2v[next(iter(wtv))])
    
    # Note: Even though it doesn't do anything, it's required that this object implement a fit method or else
    # it can't be used in a scikit-learn pipeline  
    def fit(self, X, y):
        return self
            
    def transform(self, X):
        return np.array([
            np.mean([self.w2v[w] for w in words if w in self.w2v]
                   or [np.zeros(self.dimensions)], axis=0) for words in X])




# In[ ]:

sw_list2 = sw_list = stopwords.words('english')
sw_list2 += list(string.punctuation)
sw_list2 +=  ['decision', 'use', 'claim', 'section', 'way', 'find', 'come', 'trial', 'record', 'even', 'believe','year',
 'like','want', 'argument', 'also', 'evidence', 'hold', 'ask','brief', "''", '""', '...', '``', '’', '“', '’', '”', '‘', '‘', '©',"'" 'said', 'one', 'com', 'http', '-', '–', '—', 'co', 'wa', 'ha', '1', 'amp','court', 'would', 'case', 'say', 'think', 'state', 'well', 'make','right', 'question', 'mr', 'go', 'could', 'statute', 'yes','honor', 'fact', 'justice', 'law', 'time', 'may','whether', 'take', 'get', 'act', 'know', 'point', 'issue', 'first', 'rule', 'give', 'government', 'federal', 'two', 'congress', 'judge','appeal', 'district','mean','use' 'may', 'it', 'please', 'the', 'court', 'justice', 'thank', 'you', 'mrs'] 

sw_set2 = set(sw_list2)


# In[ ]:

def process_article2(article):
    tokens = nltk.word_tokenize(article)
    stopwords_removed = [token.lower() for token in tokens if token.lower() not in sw_set2]
    return stopwords_removed 




# In[ ]:





# In[ ]:




