#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Patent Claim Scoring System
Functionality:
    Train the model only using sentence level features
@author: Zhipeng Yu
Input:  
    1.training data:'train_data_main.npy','train_y.npy'
    2.testing data:'test_data_main.npy','test_y.npy'

macOS 10.12.4
Python 3.5 NumPy 1.11.2 scikit-learn 0.18 h5py 2.6.0 Keras 1.1.0 tensorflow 0.10.0
Hardware Environment, Intel 2 GHz Intel Core i7, 8 GB 1600 MHz DDR3,
256GB SSD 
Run Time: 150s per epoch for training

Created on Sun Apr  2 17:23:24 2017

@author: Zhipeng Yu
"""

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding,Flatten
from keras.constraints import maxnorm
from keras.models import model_from_json



X_train_main=np.load("/Volumes/Zhipeng/patent_dataset/train_data_main.npy")
#X_train_add=np.load("/Volumes/Zhipeng/patent_dataset/train_data_add.npy")
X_test_main=np.load("/Volumes/Zhipeng/patent_dataset/test_data_main.npy")
#X_test_add=np.load("/Volumes/Zhipeng/patent_dataset/test_data_add.npy")
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
emb_model.add(Flatten())

#emb_model.add(Dense(32,init='normal', activation='relu',W_constraint = maxnorm(2)))
#emb_model.add(Dropout(0.2))

emb_model.add(Dense(3))
emb_model.add(Activation('softmax'))

# loss function and optimizer
emb_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
#model.fit([X_train_add,X_train], y_train, batch_size=batch_size, nb_epoch=1,
#          validation_data=([X_test_add,X_test], y_test))
hist=emb_model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10,
          )
score, acc = emb_model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
#print(hist.history)
import matplotlib.pyplot as plt
plt.plot(hist.history['acc'])
plt.ylabel('accuracy of training')
plt.xlabel('epoch')
plt.show()
'''
y_pred=emb_model.predict(X_test)
y_pred=np.argmax(y_pred, axis=1)
y_true=np.argmax(y_test,axis=1)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)

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