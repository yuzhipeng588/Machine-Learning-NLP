#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:33:40 2016

@author: Zhipeng
"""
# publish files contain original claims. patent file contains granted claims.

# Firstly,get the patent data from published file
i=0
results=[]

for line in open('pgpub_claims_fulltext.csv'):
    parts = line.split(',')
    if(i==0):
        i+=1
        continue
    #parts[0] is patent id and the first four digits indicate the year.
    #And we pick out patents from 2012 to 2013
    elif(float(parts[0])>=20120000001 and float(parts[0])<20130000002):
        #first column is patent number,then application id , claim number and claim txt.
        results.append([float(parts[0]),float(parts[1]),float(parts[2]),parts[3:len(parts)-2]])
        i +=1 
        continue

# then write the results into a txt file
with open('pub_2012to2013.txt','w')as f:
     for line in results:
         f.write(str(line[0])+','+str(line[1])+','+str(line[2])+','+','.join(line[3])+"\n")


# Repeat similar process on patent file.
i=0
patents = []
for line in open('patent_claims_fulltext.csv'):
    parts = line.split(',')
    if(i==0):
        i+=1
        continue
    try:
        float(parts[len(parts)-1])
    except ValueError:
        continue
    if float(parts[len(parts)-1])>=10468200.0 and float(parts[len(parts)-1])<=13634801.0:
        #first to last: application id,claim number,claim txt.
        patents.append([float(parts[len(parts)-1].replace("\n", "")), parts[1],parts[2:len(parts)-4]])        

# then write the results into a txt file        
with open('pat_2012to2013.txt','w')as f:
     for line in results:
         f.write(str(line[0])+','+str(line[1])+','+','.join(line[2])+"\n")

# then pair the first claim with same application id. 
dictionary={}
for line in open('pub_2012to2013.txt'):
    parts = line.split(',')
    if float(parts[2])==1.0:
        dictionary[float(parts[1])]=[','.join(parts[3:len(parts)])]

for line in open('pat2012to2013.txt'):
    parts=line.split(',')
    if float(parts[1])==1.0:
        if dictionary.has_key(float(parts[0])):
            dictionary[float(parts[0])].append(','.join(parts[2:]))

# create a new dictionary. store the paired claims
new_dic={}
for key in dictionary.keys():
    if len(dictionary[key])>1:
        new_dic[key]=dictionary[key]

# then write the results to the txt format
with open('paired_oldclaims.txt'):
    for key in new_dic.keys():
        f.write(str(key)+'\n'+new_dic[key][0]+'\n')        
with open('paired_newclaims.txt'):
    for key in new_dic.keys():
        f.write(str(key)+'\n'+new_dic[key][1]+'\n')
    



