#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:41:17 2017

@author: mac
"""

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import tensorflow as tf
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding,Flatten
from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.constraints import maxnorm
from keras.engine.topology import Merge
from sklearn.preprocessing import Imputer
from keras.models import model_from_json

#X_train=[]
#X_test=[]

X_train_main=np.load("/Volumes/Zhipeng/patent_dataset/train_data_main.npy")
X_train_add=np.load("/Volumes/Zhipeng/patent_dataset/train_data_add.npy")
X_test_main=np.load("/Volumes/Zhipeng/patent_dataset/test_data_main.npy")
X_test_add=np.load("/Volumes/Zhipeng/patent_dataset/test_data_add.npy")
y_train=np.load("/Volumes/Zhipeng/patent_dataset/train_y.npy")
y_test=np.load("/Volumes/Zhipeng/patent_dataset/test_y.npy")

max_features=20000
maxlen=500
batch_size=128

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train_main, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test_main, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
#embedding model
emb_model = Sequential()
emb_model.add(Embedding(max_features, 128, input_length=maxlen,dropout=0.2))
#emb_model.add(Convolution1D(nb_filter=32, filter_length=1, border_mode='same', activation='relu'))
#emb_model.add(MaxPooling1D(pool_length=2))

# ? may merge a layer here which include features from POST (grammar info)

#emb_model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
emb_model.add(Flatten())
#emb_model.add(Dropout(0.2))
emb_model.add(Dense(1))

#additional model
add_dim=len(X_train_add[0])
add_model = Sequential()
add_model.add(Dense(32, input_dim=add_dim, init='normal', activation='relu',W_constraint = maxnorm(2)))
add_model.add(Dense(1))
#merge model
model = Sequential()
model.add(Merge([add_model, emb_model], mode='concat', concat_axis=1))
#model.add(Dense(32))
#model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# loss function and optimizer
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
#model.fit([X_train_add,X_train], y_train, batch_size=batch_size, nb_epoch=1,
#          validation_data=([X_test_add,X_test], y_test))
model.fit([X_train_add,X_train], y_train, batch_size=batch_size, nb_epoch=3,
          )
score, acc = model.evaluate([X_test_add,X_test], y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
'''
# serialize model to JSON
model_json = model.to_json()
with open("/Users/mac/Desktop/machine_learning/firstmodel_test/4th_model_CRNN_merge2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/Users/mac/Desktop/machine_learning/firstmodel_test/4th_model_CRNN_merge2.h5")
print("Saved model to disk")
'''

'''
# later...
 
# load json and create model
json_file = open('/Users/mac/Desktop/machine_learning/firstmodel_test/4th_model_CRNN_merge2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/Users/mac/Desktop/machine_learning/firstmodel_test/4th_model_CRNN_merge2.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adm', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print "%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100)

from keras import backend as K
sess=K.get_session()
writer = tf.train.SummaryWriter("logs/", sess.graph)
# direct to the local dir and run this in terminal:
# $ tensorboard --logdir=logs
weights=model.get_weights()
'''