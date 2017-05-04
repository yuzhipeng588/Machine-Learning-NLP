#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patent Claim Scoring System
Functionality:
    Prepareing numerical training data and tesing data for the model
    Save word dictionary and reversed word dictionary for sentence level features
@author: Zhipeng Yu
Input:  
    1.raw training data :'paired_oldclaims_dep.txt','paired_newclaims_dep.txt',paired_cancledclaims_dep.txt
    2.raw testing data  :'1314paired_newclaims_dep.txt','1314paired_oldclaims_dep.txt','1314cancled_dep.txt'

Output:
    numerical training and testing data
macOS 10.12.4
Python 3.5 NumPy 1.11.2  nltk 3.2.1 collections
Hardware Environment, Intel 2 GHz Intel Core i7, 8 GB 1600 MHz DDR3,
256GB SSD 
Run Time: O(n)
Created on Mon Feb 13 15:59:41 2017

@author: Zhipeng Yu
"""
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
import re

# size of training set and testing set
train_shape=int(90000)
test_shape=int(9000)

'''
    Function:count_vectorize
    Vectorize text data based on counting pos and neg words
    Parameters:
        dictionary of word
        claim text
        maxium length
    Output:
        vectorized claim 
        
'''
def count_vectorize(dic,text,length):
    wordcount=Counter(text.split())
    wordvector=[0]*length
    for x in wordcount:
        if x in dic:
            wordvector[dic[x]-1]=wordcount[x]   
    return wordvector 
    
    
# main features from full text
    # 1.read data from txt and generate raw train and test data
i=0
X_train_new=[]
for line in open('/Volumes/Zhipeng/patent_dataset/paired_newclaims_dep.txt',encoding='utf-8',errors='ignore'):
    if i%3==1 :
        X_train_new.append([line,2])
    i+=1
    
i=0
X_test_new=[]
for line in open('/Volumes/Zhipeng/patent_dataset/1314paired_newclaims_dep.txt',encoding='utf-8',errors='ignore'):
    if i%3==1:
        X_test_new.append([line,2])
    i+=1
    
    
i=0
X_train_old=[]
for line in open('/Volumes/Zhipeng/patent_dataset/paired_oldclaims_dep.txt',encoding='utf-8',errors='ignore'):
    if i%3==1:
        X_train_old.append([line,1])
    i+=1

    
i=0
X_test_old=[]
for line in open('/Volumes/Zhipeng/patent_dataset/1314paired_oldclaims_dep.txt',encoding='utf-8',errors='ignore'):
    if i%3==1:
        X_test_old.append([line,1])
    i+=1

i=0
X_train_can=[]
for line in open('/Volumes/Zhipeng/patent_dataset/paired_cancledclaims_dep.txt',encoding='utf-8',errors='ignore'):
    if i%3==1:
        X_train_can.append([line,0])
    i+=1
    
i=0
X_test_can=[]
for line in open('/Volumes/Zhipeng/patent_dataset/1314cancled_dep.txt',encoding='utf-8',errors='ignore'):
    if i%3==1:
        X_test_can.append([line,0])
    i+=1


    # 2. shuffle the training and testing data and pick out the first 90,000 and 9,000 respectively
X_train=X_train_can+X_train_old+X_train_new
X_test=X_test_can+X_test_old+X_test_new
np.random.shuffle(X_train)
np.random.shuffle(X_test)

y_train=[item[1] for item in X_train[:train_shape]]
y_test=[item[1] for item in X_test[:test_shape]]

dic={0:[1,0,0],1:[0,1,0],2:[0,0,1]}
y_train=[dic[i] for i in y_train]
y_test=[dic[i] for i in y_test]

X_train=[item[0] for item in X_train[:train_shape]]
X_test=[item[0] for item in X_test[:test_shape]]


maxlen = 600  # cut texts after this number of words (among top max_features most common words)
batch_size = 64


    # 3. count vectorize text data
max_features = 40000
all_words = []

for text in X_train + X_test:
    all_words.extend(text.lower().split())

unique_words_ordered = [ re.sub('[^a-zA-Z]+', '', x[0]) for x in Counter(all_words).most_common() if re.sub('[^a-zA-Z]+', '', x[0]) not in stopwords.words('english') and len(re.sub('[^a-zA-Z]+', '', x[0]))>0]
word_ids = {}
rev_word_ids = {}
for i, x in enumerate(unique_words_ordered[:max_features-1]):
    word_ids[x] = i + 1  # so we can pad with 0s
    rev_word_ids[i + 1] = x

np.save('/Volumes/Zhipeng/patent_dataset/worddic.npy', word_ids) 
np.save('/Volumes/Zhipeng/patent_dataset/revdic.npy', rev_word_ids) 
    # Load
read_dictionary = np.load('/Volumes/Zhipeng/patent_dataset/revdic.npy').item()

    # 4. one hot transformation
X_train_one_hot = []
for text in X_train:
    t_ids = [word_ids[re.sub('[^a-zA-Z]+', '', x)] if re.sub('[^a-zA-Z]+', '', x) in word_ids else 0 for x in text.split()]
    item=[0]*maxlen
    item[:min(maxlen,len(t_ids))]=t_ids[:min(maxlen,len(t_ids))]
    X_train_one_hot.append(item)
    
X_test_one_hot = []
for text in X_test:
    t_ids = [word_ids[re.sub('[^a-zA-Z]+', '', x)] if re.sub('[^a-zA-Z]+', '', x) in word_ids else 0 for x in text.split()]
    item=[0]*maxlen
    item[:min(maxlen,len(t_ids))]=t_ids[:min(maxlen,len(t_ids))]
    X_test_one_hot.append(item)
    
    
# additional features from pos and neg words.

    # 1.get pos and neg words from previously processed data
negwords=[]
for line in open('/Volumes/Zhipeng/patent_dataset/negword_freq.csv'):
    parts=line.lower().split(',')
    negwords.append(parts[0])

poswords=[]
for line in open('/Volumes/Zhipeng/patent_dataset/posword_freq.csv'):  
    parts=line.lower().split(',')
    poswords.append(parts[0])
    
    # 2.build dictionary for count vectorize
negword_ids = {}
negrev_word_ids = {}
for i, x in enumerate(negwords):
    negword_ids[x] = i + 1  # so we can pad with 0s
    negrev_word_ids[i + 1] = x   

posword_ids = {}
posrev_word_ids = {}
for i, x in enumerate(poswords):
    posword_ids[x] = i + 1  # so we can pad with 0s
    posrev_word_ids[i + 1] = x

    # 3.get additional features
X_train_add=[]
for row in X_train:
    neg_vector=count_vectorize(negword_ids,row,len(negwords))
    pos_vector=count_vectorize(posword_ids,row,len(poswords))
    X_train_add.append(neg_vector+pos_vector)

X_test_add=[]
for row in X_test:
    neg_vector=count_vectorize(negword_ids,row,len(negwords))
    pos_vector=count_vectorize(posword_ids,row,len(poswords))
    X_test_add.append(neg_vector+pos_vector)


np.save("/Volumes/Zhipeng/patent_dataset/train_data_main", X_train_one_hot)
np.save("/Volumes/Zhipeng/patent_dataset/test_data_main", X_test_one_hot)

np.save("/Volumes/Zhipeng/patent_dataset/train_data_add", X_train_add)
np.save("/Volumes/Zhipeng/patent_dataset/test_data_add", X_test_add)

np.save("/Volumes/Zhipeng/patent_dataset/train_y", y_train)
np.save("/Volumes/Zhipeng/patent_dataset/test_y", y_test)
