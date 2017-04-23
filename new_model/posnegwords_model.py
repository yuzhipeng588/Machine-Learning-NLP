#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:57:16 2017

@author: mac
"""


from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import tensorflow as tf
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import csv
from keras.constraints import maxnorm

from keras.models import model_from_json

#X_train=[]
#X_test=[]

X_train_add=np.load("/Volumes/Zhipeng/patent_dataset/train_data_add.npy")
X_test_add=np.load("/Volumes/Zhipeng/patent_dataset/test_data_add.npy")
y_train=np.load("/Volumes/Zhipeng/patent_dataset/train_y.npy")
y_test=np.load("/Volumes/Zhipeng/patent_dataset/test_y.npy")

max_features=40000
#maxlen=600
batch_size=128

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train_add)
X_test = sequence.pad_sequences(X_test_add)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')

#additional model
add_dim=len(X_train_add[0])
add_model = Sequential()
#add_model.add(Dense(32, input_dim=add_dim, init='normal', activation='relu',W_constraint = maxnorm(2)))
#add_model.add(Dropout(0.2))
#add_model.add(Dense(3))
add_model.add(Dense(3,input_dim=add_dim))
add_model.add(Activation('softmax'))

# loss function and optimizer
add_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
#model.fit([X_train_add,X_train], y_train, batch_size=batch_size, nb_epoch=1,
#          validation_data=([X_test_add,X_test], y_test))
hist=add_model.fit(X_train_add, y_train, batch_size=batch_size, nb_epoch=5
          )
score, acc = add_model.evaluate(X_test_add, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
#print(hist.history)
import matplotlib.pyplot as plt
plt.plot(hist.history['acc'])
plt.ylabel('accuracy of training')
plt.xlabel('epoch')
plt.show()



weights=[]
for layer in add_model.layers:
    weights.append(layer.get_weights())
weightall= np.array(weights[0][0])
weightall=np.reshape(weightall,weightall.shape[0])

weight_sorted=np.argsort(weightall)
weight_sortedvalue=np.sort(weightall)

negwords=[]
for line in open('/Volumes/Zhipeng/patent_dataset/negword_freq.csv'):
    parts=line.lower().split(',')
    negwords.append(parts[0])

poswords=[]
for line in open('/Volumes/Zhipeng/patent_dataset/posword_freq.csv'):  
    parts=line.lower().split(',')
    poswords.append(parts[0])
    
negposwords=negwords+poswords
negwords_sorted=[[negposwords[i],weight_sortedvalue[index]] for index,i in enumerate(weight_sorted) if i<len(negwords)]
poswords_sorted=[[negposwords[i],weight_sortedvalue[::-1][index]] for index,i in enumerate(weight_sorted[::-1]) if i<len(negwords)]

f = open("/Volumes/Zhipeng/patent_dataset/negword_importances_weight.csv", 'w',newline='')
writer = csv.writer(f)
writer.writerow( ('NegWord', 'Weight' ))  
i=0
for k in negwords_sorted:
    writer.writerow((k[0],k[1]))
    i+=1
f.close()

f = open("/Volumes/Zhipeng/patent_dataset/posword_importances_weight.csv", 'w',newline='')
writer = csv.writer(f)
writer.writerow( ('PosWord', 'Weight' ))  
i=0
for k in poswords_sorted:
    writer.writerow((k[0],k[1]))
    i+=1
f.close()

f = open("/Volumes/Zhipeng/patent_dataset/negword_importances.csv", 'w',newline='')
writer = csv.writer(f)
writer.writerow( ('NegWord', 'Rank' ))  
i=1 
for k in negwords_sorted:
    writer.writerow((k[0],i))
    i+=1
f.close()

f = open("/Volumes/Zhipeng/patent_dataset/posword_importances.csv", 'w',newline='')
writer = csv.writer(f)
writer.writerow( ('PosWord', 'Rank' ))  
i=1 
for k in poswords_sorted:
    writer.writerow((k[0],i))
    i+=1
f.close()

'''
# serialize model to JSON
add_model_json = add_model.to_json()
with open("/Users/mac/Machine-Learning-NLP/DataEngineering/PosnegOnly.json", "w") as json_file:
    json_file.write(add_model_json)
# serialize weights to HDF5
add_model.save_weights("/Users/mac/Machine-Learning-NLP/DataEngineering/PosnegOnly.h5")
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