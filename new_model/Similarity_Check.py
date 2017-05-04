#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patent Claim Scoring System
Functionality:
    Calculate similarity matrix
@author: Zhipeng Yu
Input:  
    granted claims published in 2012:'paired_newclaims_dep.txt'

Limitaion: need large memority to store the whole similarity matrix
macOS 10.12.4
Python 3.5 NumPy 1.11.2 scikit-learn 0.18 
Hardware Environment, Intel 2 GHz Intel Core i7, 8 GB 1600 MHz DDR3,
256GB SSD 
Created on Fri Jan 27 00:21:37 2017

@author: mac
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
import numpy as np
i=0
y_train=[]
for line in open('/Volumes/Zhipeng/patent_dataset/paired_newclaims_2012to2013.txt'):
    if i%2==0:
        y_train.append(line)
    i+=1

vectorizer = TfidfVectorizer(min_df=1)
X=vectorizer.fit_transform(y_train)
#pca = decomposition.PCA(n_components=1)
#pca.fit(X)
#data_reduced_to_one_dimension = pca.transform(X)
#sim=(X * X.T).A


# because of the limitation, in this code we only show similarity scores for only one claim
sim0=(X[0,:]*X.T).A
top10=np.argsort(sim0)[0][-10:]