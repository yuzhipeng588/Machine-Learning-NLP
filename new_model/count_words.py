#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:58:46 2017

@author: mac
"""
from nltk.corpus import stopwords
import re
from nltk.corpus import words
import csv
stop = set(stopwords.words('english'))
wordcount={}
for line in open('/Volumes/Zhipeng/patent_dataset/paired_newclaims_2012to2013.txt'):
    for i in line.lower().split():
        i=re.sub('[^a-zA-Z]+', '', i)
        if i not in stop and i in words.words():
            if i not in wordcount:
                wordcount[i] = 1
            else:
                wordcount[i] += 1
for line in open('/Volumes/Zhipeng/patent_dataset/paired_oldclaims_2012to2013.txt'):
    for i in line.lower().split():
        i=re.sub('[^a-zA-Z]+', '', i)
        if i not in stop and i in words.words():
            if i not in wordcount:
                wordcount[i] = -1
            else:
                wordcount[i] -= 1
wordcount_cleaned={}
wordcount_cleaned = {k: v for k, v in wordcount.items() if abs(v) > 10}

negword={}
negword={k:v for k,v in wordcount_cleaned.items() if v<0}

posword={}
posword={k:v for k,v in wordcount_cleaned.items() if v>0}

f = open("negword_freq", 'w',newline='')
writer = csv.writer(f)
writer.writerow( ('NegWord', 'Frequency' ))   
for (k,v) in negword.items():
    writer.writerow((k,v))
f.close()

f = open("posword_freq", 'w',newline='')
writer = csv.writer(f)
writer.writerow( ('PosWord', 'Frequency' ))   
for (k,v) in posword.items():
    writer.writerow((k,v))
f.close()
        
    