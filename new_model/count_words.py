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
for line in open('/Volumes/Zhipeng/patent_dataset/paired_newclaims_dep.txt',encoding='utf-8',errors='ignore'):
    for i in line.lower().split():
        i=re.sub('[^a-zA-Z]+', '', i)
# too slow if check i in words() or not
#        if i not in stop and i in words.words(): 
        if i not in stop and len(i)>0:
            if i not in wordcount:
                wordcount[i] = 1
            else:
                wordcount[i] += 1
for line in open('/Volumes/Zhipeng/patent_dataset/paired_oldclaims_dep.txt'):
    for i in line.lower().split():
        i=re.sub('[^a-zA-Z]+', '', i)
#        if i not in stop and i in words.words():
        if i not in stop and len(i)>0:
            if i not in wordcount:
                wordcount[i] = -1
            else:
                wordcount[i] -= 1


negword={}
negword={k:v for k,v in wordcount.items() if v<-10}

posword={}
posword={k:v for k,v in wordcount.items() if v>1000}

f = open("/Volumes/Zhipeng/patent_dataset/negword_freq.csv", 'w',newline='')
writer = csv.writer(f)
writer.writerow( ('NegWord', 'Frequency' ))   
for (k,v) in negword.items():
    writer.writerow((k,v))
f.close()

f = open("/Volumes/Zhipeng/patent_dataset/posword_freq.csv", 'w',newline='')
writer = csv.writer(f)
writer.writerow( ('PosWord', 'Frequency' ))   
for (k,v) in posword.items():
    writer.writerow((k,v))
f.close()
        
    