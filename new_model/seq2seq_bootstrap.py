#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:44:53 2017

@author: mac
"""

import numpy as np
import tensorflow as tf
import tempfile



batch_size = 64
seq_length = batch_size

# vocab_size need +1 for " "
vocab_size = 20000
embedding_dim = 100

memory_dim = 100
i=0
y_train=[]
#for line in open('/Volumes/Zhipeng/patent_dataset/paired_newclaims_2012to2013.txt'):
for line in open('paired_newclaims_2012to2013.txt'):
    if i%2==0:
        y_train.append(line)
    i+=1
i=0
x_train=[]
#for line in open('/Volumes/Zhipeng/patent_dataset/paired_oldclaims_2012to2013.txt'):
for line in open('paired_oldclaims_2012to2013.txt'):
    if i%2==0:
        x_train.append(line)
    i+=1



from collections import Counter
max_len = seq_length
max_features = vocab_size
all_words = []

for text in x_train+y_train:
    all_words.extend(text.split())
unique_words_ordered = [x[0] for x in Counter(all_words).most_common()]
word_ids = {}
rev_word_ids = {}
for i, x in enumerate(unique_words_ordered[:max_features-1]):
    word_ids[x] = i + 1  
    rev_word_ids[i + 1] = x

# need consider len < 200
X_train_one_hot = []
for text in x_train:
    t_ids = [word_ids[x] for x in text.split() if x in word_ids]
    if len(t_ids)>=max_len:
        t_ids=t_ids[0:max_len]
        X_train_one_hot.append(t_ids)
    else:
        tlen=len(t_ids)
        for i in range(max_len-tlen):t_ids.append(0) 
        X_train_one_hot.append(t_ids)
Y_train_one_hot = []
for text in y_train:
    t_ids = [word_ids[x] for x in text.split() if x in word_ids]
    if len(t_ids)>=max_len:
        t_ids=t_ids[0:max_len]
        Y_train_one_hot.append(t_ids)
    else:
        tlen=len(t_ids)
        for i in range(max_len-tlen):t_ids.append(0) 
        Y_train_one_hot.append(t_ids)
    

graph = tf.Graph()
    
with graph.as_default():
    enc_inp = [tf.placeholder(tf.int32, shape=(None,),name="inp%i" % t) for t in range(seq_length)]
    labels = [tf.placeholder(tf.int32, shape=(None,),name="labels%i" % t) for t in range(seq_length)]
    weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels]
# Decoder input: prepend some "GO" token and drop the final
# token of the encoder input
    dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32, name="GO")] + enc_inp[:-1])
# Initial memory value for recurrence.
    prev_mem = tf.zeros((batch_size, memory_dim))
    cell = tf.nn.rnn_cell.GRUCell(memory_dim)
    dec_outputs, dec_memory = tf.nn.seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, vocab_size, vocab_size,embedding_size=embedding_dim)
    loss = tf.nn.seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)
    tf.scalar_summary("loss", loss)
    magnitude = tf.sqrt(tf.reduce_sum(tf.square(dec_memory[1])))
    tf.scalar_summary("magnitude at t=1", magnitude)
    summary_op = tf.merge_all_summaries()
    learning_rate = 0.05
    momentum = 0.9
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    train_op = optimizer.minimize(loss)


    
    
num_steps = 1001
with tf.Session(graph=graph) as session:
  session.run(tf.initialize_all_variables())
  print('Initialized')
  results1=[]
  results2=[]
  for step in range(num_steps):
      offset = (step * batch_size) % (len(Y_train_one_hot) - batch_size)
      batch_data = X_train_one_hot[offset:(offset + batch_size)]
      batch_labels = Y_train_one_hot[offset:(offset + batch_size)]
      feed_dict = {enc_inp[t]: batch_data[t] for t in range(seq_length)}
      feed_dict.update({labels[t]: batch_labels[t] for t in range(seq_length)})
      dec_outputs_batch = session.run(dec_outputs, feed_dict)
      if step==num_steps-1:
          results1=batch_data
          results2=batch_labels
#    output, loss_t, summary,dec_outputs_batch = session.run([train_op, loss, summary_op,dec_outputs], feed_dict)
#save the session
  saver = tf.train.Saver()
  saver.save(session, 'my-model')
  meta_graph_def = tf.train.export_meta_graph(filename='my-model.meta')
  '''
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  '''
  #print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
  out_idx=[logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]
  out_text=[]
  for row in out_idx:
      one_row=[]
      for i in row:
          one_row.append(rev_word_ids[i])
      out_text.append(one_row)
  a=" ".join(out_text[0])
#restore session
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('my-model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    feed_dict={enc_input[t]:X_train_one_hot[t] for t in range(seq_length)}
    feed_dict.update({labels[t]:Y_train_one_hot[t] for t in range(seq_length)})
    dec_outputs_batch = session.run(dec_outputs, feed_dict)