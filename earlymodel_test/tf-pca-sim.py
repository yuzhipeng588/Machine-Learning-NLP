#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 00:21:37 2017

@author: mac
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
i=0
y_train=[]
for line in open('/Volumes/Zhipeng/patent_dataset/1314paired_newclaims_dep.txt',encoding='utf-8',errors='ignore'):
    if i%3==0:
        y_train.append([line])
    elif i%3==1:
        y_train[-1].append(line)
    i+=1
'''   
i=0
x_train=[]
for line in open('/Volumes/Zhipeng/patent_dataset/paired_oldclaims_2012to2013.txt'):
    if i%2==0:
        x_train.append(line)
    i+=1
'''
vectorizer = TfidfVectorizer(min_df=1)
X=vectorizer.fit_transform([item[1] for item in y_train])
#pca = decomposition.PCA(n_components=1)
#pca.fit(X)
#data_reduced_to_one_dimension = pca.transform(X)
#sim=(X * X.T).A


#subX=X[0:100,:]
#sub_sim=(subX * subX.T).A

sim0=(X[1,:]*X.T).A
top10_value=np.sort(sim0)[0][-10:][::-1]
top10=np.argsort(sim0)[0][-10:][::-1]

top10_content=[y_train[i] for i in top10]
[y_train[i][0] for i in top10]