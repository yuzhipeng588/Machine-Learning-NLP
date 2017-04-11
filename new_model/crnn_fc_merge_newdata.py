#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:03:41 2017

@author: mac
"""
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import tensorflow as tf
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU,Flatten
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

max_features=40000
maxlen=600
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
#emb_model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
#emb_model.add(MaxPooling1D(pool_length=2))

# ? may merge a layer here which include features from POST (grammar info)

#emb_model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
emb_model.add(Flatten())
#emb_model.add(Dense(32))
#additional model
add_dim=len(X_train_add[0])
add_model = Sequential()
add_model.add(Dense(32, input_dim=add_dim, init='normal', activation='relu',W_constraint = maxnorm(2)))

#merge model
model = Sequential()
model.add(Merge([add_model, emb_model], mode='concat', concat_axis=1))
#model.add(Dense(32))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# loss function and optimizer
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
#model.fit([X_train_add,X_train], y_train, batch_size=batch_size, nb_epoch=1,
#          validation_data=([X_test_add,X_test], y_test))
hist=model.fit([X_train_add,X_train], y_train, batch_size=batch_size, nb_epoch=3,
          )
score, acc = model.evaluate([X_test_add,X_test], y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

'''
from keras import backend as K

# get output after embedding layer 
# learning_phase:0 means output is obtained from test process as there is a dropout(train and test have different processes)
get_3rd_layer_output = K.function([emb_model.layers[0].input,K.learning_phase()],
                                  [emb_model.layers[0].output])
layer_output = get_3rd_layer_output([X_test,0])[0]

#get weight
weights=[]
for layer in model.layers:
    weights.append(layer.get_weights())
    
#weights[1] is from final layer dense(1)(weights and bias) we only interest in weights
#the last maxlen*emb_dimention entries are for embedding features
weight_emb=weights[1][0][-maxlen*128:]

# weight * output of embedding layer 
example=layer_output[0,:,:]
words_weights=[]
for index, item in enumerate(example):
    words_weights.append(np.dot(item,weight_emb[index*128:min(len(weight_emb),index*128+128)]))
    
words_weights=np.array(words_weights)
words_weights=np.reshape(words_weights,words_weights.shape[0])

# sort and know where is important for a given claim
weights_sorted=np.sort(words_weights)
words_sorted=np.argsort(words_weights)
text=X_test[0]

# (sorted from negative to positive) The impact on a given claim
words_pos=text[words_sorted]
words=[rev_word_ids[i] for i in words_pos]
'''


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