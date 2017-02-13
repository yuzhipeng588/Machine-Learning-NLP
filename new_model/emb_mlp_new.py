#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 17:49:55 2017

@author: mac
"""

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding,Flatten,Reshape
from keras.layers import LSTM, SimpleRNN, GRU
from keras.constraints import maxnorm
import numpy as np

# read data from txt and generate raw train and test data
i=0
X_train_new=[]
for line in open('/Volumes/Zhipeng/patent_dataset/paired_newclaims_2012to2013.txt'):
    if i%2==0:
        X_train_new.append(line)
    i+=1
X_test_new=X_train_new[-1000:]   
X_train_new=X_train_new[:10000]
i=0
X_train_old=[]
for line in open('/Volumes/Zhipeng/patent_dataset/paired_oldclaims_2012to2013.txt'):
    if i%2==0:
        X_train_old.append(line)
    i+=1
X_test_old=X_train_old[-1000:]
X_train_old=X_train_old[:10000]

X_train=X_train_old+X_train_new
X_test=X_test_old+X_test_new

y_train = np.append(np.zeros(len(X_train_old)),np.ones(len(X_train_new)))
y_test = np.append(np.zeros(len(X_test_old)),np.ones(len(X_test_new)))

maxlen = 200  # cut texts after this number of words (among top max_features most common words)
batch_size = 64

from collections import Counter

max_features = 20000
all_words = []

for text in X_train + X_test:
    all_words.extend(text.split())
unique_words_ordered = [x[0] for x in Counter(all_words).most_common()]
word_ids = {}
rev_word_ids = {}
for i, x in enumerate(unique_words_ordered[:max_features-1]):
    word_ids[x] = i + 1  # so we can pad with 0s
    rev_word_ids[i + 1] = x

X_train_one_hot = []
for text in X_train:
    t_ids = [word_ids[x] for x in text.split() if x in word_ids]
    X_train_one_hot.append(t_ids)
    
X_test_one_hot = []
for text in X_test:
    t_ids = [word_ids[x] for x in text.split() if x in word_ids]
    X_test_one_hot.append(t_ids)

    
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train_one_hot, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test_one_hot, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
embedding_size=128
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen,dropout=0.2))
print('first layer...')
#model.add(Reshape((1, maxlen, embedding_size)))
model.add(Flatten())
'''
model.add(Dense(1000, input_shape=(embedding_size*max_features,)))
print('seconde layer...')
model.add(Dropout(0.2))
'''
model.add(Dense(500, init='normal', activation='relu',W_constraint = maxnorm(2)))
model.add(Dropout(0.2))
model.add(Dense(200, init='normal', activation='relu',W_constraint = maxnorm(2)))
model.add(Dropout(0.2))
model.add(Dense(200, init='normal', activation='relu',W_constraint = maxnorm(2)))
model.add(Dropout(0.2))
model.add(Dense(100, init='normal', activation='relu',W_constraint = maxnorm(2)))
model.add(Dropout(0.2))
model.add(Dense(1, init='normal', activation='sigmoid',W_constraint = maxnorm(2)))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)