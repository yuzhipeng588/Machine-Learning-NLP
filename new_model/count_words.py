#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patent Claim Scoring System
Functionality:
    Creat negative and positive word list by comparing claims in initial stage with final stage
@author: Zhipeng Yu
Input:  
    1.paired data (initial version of claims):'paired_oldclaims_dep.txt'
    2.paired data (final version of claims):'paired_newclaims_dep.txt'

Output:
    negative and positive word list
macOS 10.12.4
Python 3.5 NumPy 1.11.2  nltk 3.2.1
Hardware Environment, Intel 2 GHz Intel Core i7, 8 GB 1600 MHz DDR3,
256GB SSD 
Run Time: O(n)

Created on Fri Feb 10 15:58:46 2017

@author: Zhipeng Yu
"""
from nltk.corpus import stopwords
import re
from nltk.corpus import words
import csv
stop = set(stopwords.words('english'))
wordcount={}

# count words: deleted words -1 , added words +1 
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
for line in open('/Volumes/Zhipeng/patent_dataset/paired_oldclaims_dep.txt',encoding='utf-8',errors='ignore'):
    for i in line.lower().split():
        i=re.sub('[^a-zA-Z]+', '', i)
#        if i not in stop and i in words.words():
        if i not in stop and len(i)>0:
            if i not in wordcount:
                wordcount[i] = -1
            else:
                wordcount[i] -= 1

# make sure each word is an english word
negword={}
negword={k:v for k,v in wordcount.items() if v<-1 and k in words.words()}

posword={}
posword={k:v for k,v in wordcount.items() if v>1000 and k in words.words()}

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
        
    