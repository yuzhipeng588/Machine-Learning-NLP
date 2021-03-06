#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patent Claim Scoring System
Functionality:
    Pair granted claims between publications and patents datasets 
    Get three types of claims:
        1.cancled
        2.granted after modifications
        3.granted directly
@author: Zhipeng Yu
Input:'pgpub_claims_fulltext.csv' and 'pub_2013to2014_dep.txt'
        (available in https://www.uspto.gov/learning-and-resources/electronic-data-products/patent-claims-research-dataset)
Limitaion: need large memory
Linux Kernel (3.12.2) on Lee Flemaing's lab server
Python 2.7 NumPy 1.12.1
Hardware Environment, Intel 2 GHz Intel Core i7, 8 GB 1600 MHz DDR3,
256GB SSD 
Run Time: O(n) because it needs to traverse the whole file once. 

Created on Wed Mar  1 00:33:26 2017

@author: Zhipeng Yu
"""

# open publication file and extract application id, text content, dependent claims for any 
# independent claim in 2013 to 2014
i=0
results=[]
patID=0.0
each=[]
for line in open('pgpub_claims_fulltext.csv'):
    parts = line.split(',')
    if(i==0):
        i+=1
        continue
    #parts[0] is patent id and the first four digits indicate the year.
    #And we pick out patents from 2013 to 2014
    try:
        a=float(parts[0])
    except ValueError:
        continue
#    if(float(parts[0])>=20120000001 and float(parts[0])<20130000002):
    if(float(parts[0])>=20130000001 and float(parts[0])<20140000002):
        #first column is patent number,then application id , claim txt and dependencies text.
        if patID!=float(parts[0].replace("\n", "")):
            patID=float(parts[0].replace("\n", ""))
            if len(each)!=0:
                results.append(each)
            each=[]
        # only keep claim 1
        try:
            a=float(parts[2])
        except ValueError:
            continue
        if float(parts[2].replace("\n", ""))==1.0:
            each.insert(0,parts[3:len(parts)-2])
            each.insert(0,parts[1])
            each.insert(0,parts[0])
        elif 'claim 1 ' in parts[3:len(parts)-2]:
            each.append(parts[3:len(parts)-2])

# then write the results into a txt file
with open('pub_2013to2014_dep.txt','w')as f:
     for line in results:
         if len(line[2])==0:continue
         content=str(line[0])+','+str(line[1])+','+','.join(line[2])+"\t"
         for element in line[3:]:
             content=content+','.join(element)+"\t"
         content=content+'\n'
         f.write(content)


# Repeat similar process on patent file.
i=0
patents = []
appID=0
each=[]
for line in open('patent_claims_fulltext.csv'):
    parts = line.split(',')
    if(i==0):
        i+=1
        continue
    try:
        float(parts[len(parts)-1])
    except ValueError:
        continue
#    if float(parts[len(parts)-1])>=10468200.0 and float(parts[len(parts)-1])<=13634801.0:
    if float(parts[len(parts)-1])>=9962981 and float(parts[len(parts)-1])<=14023933:
        #first to last: application id,claim number,claim txt.
        if appID!=float(parts[len(parts)-1]):
            appID=float(parts[len(parts)-1])
            if len(each)!=0:
                patents.append(each)
            each=[]
        try:
            float(parts[1])
        except ValueError:
            continue
        if float(parts[1].replace("\n", ""))==1.0:
            each.insert(0,parts[2:len(parts)-3])
            each.insert(0,float(parts[len(parts)-1].replace("\n", "")))
        elif 'claim 1 ' in str(','.join(parts[2:len(parts)-3])) or 'claim 1,' in str(','.join(parts[2:len(parts)-3])) or 'claim 1.' in str(','.join(parts[2:len(parts)-3])):
            each.append(parts[2:len(parts)-3])

# then write the results into a txt file        
with open('pat_2013to2014_dep.txt','w')as f:
     for line in patents:
         if len(line)==1: print(line)
         if len(line)<=1 or len(line[1])==0:continue
         content=str(line[0])+','+','.join(line[1])+"\t"
         for element in line[2:]:
             content=content+','.join(element)+"\t"
         content=content+'\n'
         f.write(content)

# then pair the first claim with same application id. 
dictionary={}
for line in open('pub_2012to2013_dep.txt'):
    parts = line.split(',')
    dictionary[float(parts[1])]=[','.join(parts[2:len(parts)])]

for line in open('pat_2012to2013_dep.txt'):
    parts=line.split(',')
    if dictionary.has_key(float(parts[0])):
        dictionary[float(parts[0])].append(','.join(parts[1:]))

'''        
for line in patents:
     try:
             float(line[0])
     except ValueError and TypeError:
             continue
     if dictionary.has_key(float(line[0])):
             dictionary[float(line[0])].append('\t'.join([','.join(i) for i in line[1:] if len(i)>0 and ('claim 1.' in ','.join(i) or 'claim 1,' in ','.join(i) or 'claim 1 ' in ','.join(i) or '"1.' in ','.join(i))]))
'''        
        
# create a new dictionary. store the paired claims
new_dic={}
for key in dictionary.keys():
    if len(dictionary[key])>1 :#and len(dictionary[key][0])==len(dictionary[key][1]):
        if dictionary[key][0]!=dictionary[key][1]:
            new_dic[key]=dictionary[key]

# then write the results to the txt format
with open('1314paired_oldclaims_dep.txt','w')as f:
    for key in new_dic.keys():
        f.write(str(key)+'\n'+new_dic[key][0]+'\n')  
        
with open('1314paired_newclaims_dep.txt','w') as f:
    for key in new_dic.keys():
        f.write(str(key)+'\n'+new_dic[key][1]+'\n'+'\n')

with open('1314cancled_dep.txt','w') as f:
    for key in dictionary.keys():
        if len(dictionary[key])==1:
            f.write(str(key)+'\n'+dictionary[key][0]+'\n')