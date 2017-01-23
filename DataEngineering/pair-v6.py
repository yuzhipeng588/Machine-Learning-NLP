#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:33:40 2016

@author: mac
"""

import pandas as pd
import numpy as np
import codecs
i=0
patent=[]

for line in open('publications_total.csv'):
    parts = line.split(',')
    if(i==0):
        i+=1
        continue
    if(float(parts[0])>=20010000001 and float(parts[0])<20020000002):
        patent.append([parts[0],parts[1],parts[3]])
        i+=1
        continue
publish=np.array(patent)
publish=publish.astype(float)
patent=[]
i=0
lower=np.min(publish[:,1])#10461848
upper=np.max(publish[:,1])#14116425

pat_file=codecs.open('/Users/mac/Desktop/machine_learning/firstmodel_test/patent_claims_fulltext.csv', "r",encoding='utf-8', errors='ignore')
for line in pat_file:
    parts = line.split(',')
    if(i==0):
        i+=1
        continue
    try:
        float(parts[len(parts)-1])
    except ValueError:
        continue
    if(float(parts[len(parts)-1])>=7153880 and float(parts[len(parts)-1])<=9939815):
        patent.append([parts[len(parts)-1],parts[2:len(parts)-3]])
patent=np.array(patent)
patent=patent.astype(float)        
publish_df= pd.DataFrame(publish,columns = ['pub_no','appl_id'])
patents_df = pd.DataFrame(patent,columns = ['appl_id'])
patents_df['appl_id'] = patents_df['appl_id'].astype(int)
publish_df['appl_id'] = publish_df['appl_id'].astype(int)
set1=patents_df['appl_id'].tolist()
set2 = np.column_stack((publish_df['appl_id'].tolist(), publish_df['pub_no'].tolist()))
#set1.sort()
#set2.sort()
Data = pd.merge(patents_df, publish_df, on='appl_id', how='inner')
'''
paired_pubno=[]
for item1 in set1:
    for item2_ap,item2_pub in set2:
        if item1==item2_ap:
            paired_pubno.append(item2_pub)
'''       