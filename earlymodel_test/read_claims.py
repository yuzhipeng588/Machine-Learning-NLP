#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:36:44 2016

@author: Yu Zhipeng
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC
import pandas as pd

pinitial_train=pd.read_csv('/Volumes/Zhipeng/patent_dataset/pgpub_claims_fulltext.csv',delimiter='\t',nrows=3000,encoding='utf-8')
pfinal_train=pd.read_csv('/Volumes/Zhipeng/patent_dataset/patent_claims_fulltext.csv',delimiter='\t',nrows=3000,encoding='utf-8')
pinitial_test=pd.read_csv('/Volumes/Zhipeng/patent_dataset/pgpub_claims_fulltext.csv',delimiter='\t',nrows=3000,skiprows=range(1,30000),encoding='utf-8')
pfinal_test=pd.read_csv('/Volumes/Zhipeng/patent_dataset/patent_claims_fulltext.csv',delimiter='\t',nrows=3000,skiprows=range(1,30000),encoding='utf-8')
import numpy as np
X_train = pinitial_train['pub_no,appl_id,claim_no,claim_txt,dependencies,ind_flg'].tolist()+pfinal_train['pat_no,claim_no,claim_txt,dependencies,ind_flg,appl_id'].tolist()#.astype(str)
y_train = np.append(np.zeros(len(pinitial_train)),np.ones(len(pfinal_train)))

X_test = pinitial_test['pub_no,appl_id,claim_no,claim_txt,dependencies,ind_flg'].tolist()+pfinal_test['pat_no,claim_no,claim_txt,dependencies,ind_flg,appl_id'].tolist()
y_test = np.append(np.zeros(len(pinitial_test)),np.ones(len(pfinal_test)))
'''
X_test = data["test"]["pos"] + data["test"]["neg"]
y_test = np.append(np.ones(_len(data["test"]["pos"])), np.zeros(len(data["test"]["neg"])))
'''
print(len(X_train), len(y_train))
print(len(X_test), len(y_test))

# For countvect and tfidf linear model
#tfidf transformation is automately employed in Pipeline
## tfidf = TfidfVectorizer()
## tfidf.fit_transform(X_train)

# build a pipeline - SVC
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                    ('tfidf', TfidfTransformer()),
                    ('clf', OneVsRestClassifier(LinearSVC(random_state=0)))
                     ])

# fit using pipeline
clf = text_clf.fit(X_train, y_train)
# predict
#predicted = clf.predict(X_test)
clf.score(X_train, y_train) 
clf.score(X_test, y_test) 
# print metrics
#print(metrics.classification_report(y_test, predicted))

'''
vectorizer = CountVectorizer(ngram_range=(1, 2))
vect_fea_train=vectorizer.fit_transform(X_train)
vect_fea_test=vectorizer.fit_transform(X_test)
tf_transformer = TfidfTransformer(smooth_idf=False)
input_train=tf_transformer.fit_transform(vect_fea_train)
input_test=tf_transformer.fit_transform(vect_fea_test)
'''