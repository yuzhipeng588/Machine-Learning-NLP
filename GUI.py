#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:15:32 2017

@author: mac
"""
from __future__ import print_function
import tkinter
from tkinter import ttk,Text
from collections import Counter
import numpy as np
import re
from keras import backend as K

maxlen=600


class Adder(ttk.Frame):
    """The adders gui and functions."""
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.pos_words = []
        self.init_gui()

    def on_quit(self):
        """Exits program."""
        quit()
    
        
    # vectorize text data based on counting pos and neg words
    def count_vectorize(self,dic,text,length):
            wordcount=Counter(text.split())
            wordvector=[0]*length
            for x in wordcount:
                if x in dic:
                    wordvector[dic[x]-1]=wordcount[x]   
            return wordvector    
            
    def calculate(self):
        """Calculates the score and hightlight pos and neg words."""
        # get input from text box
        inputvalue = self.textbox.get("1.0",'end-1c')
        inputvalue = ''.join(inputvalue)
        
        # extract features from raw text
        X_add = np.array(self.get_addData(inputvalue))
        X_add = X_add.reshape((1,len(X_add)))
        X_main = np.array(self.get_mainData(inputvalue))
        X_main = X_main.reshape((1,len(X_main)))
        # predict 
        score = self.model.predict([X_add,X_main])
        # display
        self.answer_label['text'] = score
        
        # get pos and words
        self.pos_words,self.neg_words = self.get_weightedWords(X_main)
        
        # store all the lengths
        words = inputvalue.split()
        words_length = [0]
        for i in words:
            # +1 for space separator
            words_length.append(words_length[-1]+len(i)+1)

        #  draw color to pos words
        if len(self.pos_words) == 0:
            self.textbox.tag_add("none", "1.0", 'end-1c')
            self.textbox.tag_config("none", background="white", foreground="black")
        else:
            # track the pos words and get its line.column expression
            numChar_before = [words_length[i] for i in self.pos_words]
            numChar_after =  [words_length[i+1] for i in self.pos_words]
            for i in range (len(numChar_before)):
                beg = self.getTextIndex(numChar_before[i]+1)
                end = self.getTextIndex(numChar_after[i])
                print(beg,end)
                if len(beg)!=0 and len(end)!=0:
                    self.textbox.tag_add("pos", beg, end)
                    self.textbox.tag_config("pos", background="red", foreground="blue")
        
        #  draw color to neg words
        if len(self.neg_words) == 0:
            self.textbox.tag_add("none", "1.0", 'end-1c')
            self.textbox.tag_config("none", background="while", foreground="black")
        else:
            # track the pos words and get its line.column expression
            numChar_before = [words_length[i] for i in self.neg_words]
            numChar_after =  [words_length[i+1] for i in self.neg_words]
            for i in range (len(numChar_before)):
                beg = self.getTextIndex(numChar_before[i]+1)
                end = self.getTextIndex(numChar_after[i])
                print(beg,end)
                if len(beg)!=0 and len(end)!=0:
                    self.textbox.tag_add("neg", beg, end)
                    self.textbox.tag_config("neg", background="green", foreground="yellow")

    # get line.column expression given the index position of the words
    def getTextIndex(self,num):
        pre = 0
        # +1 because line start from 1 not 0
        for i in range(int(self.textbox.index('end-1c').split('.')[0])+1):
            line_len = int(self.textbox.index('%d.end' % (i+1)).split('.')[1])
            if num <= pre + line_len:
                return str(i+1)+'.'+str(num-pre-1)
            else:
                pre += line_len
            
        return ''
    
    # get index position of pos words
    def get_weightedWords(self,X_test):
        print(X_test)
        # get output after embedding layer 
        # learning_phase:0 means output is obtained from test process as there is a dropout(train and test have different processes)
        get_3rd_layer_output = K.function([self.emb_model.layers[0].input,K.learning_phase()],
                                          [self.emb_model.layers[0].output])
        layer_output = get_3rd_layer_output([X_test,0])[0]
        
        #get weight
        weights=[]
        for layer in self.model.layers:
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
        words_sorted=np.argsort(words_weights)

        # return postion of pos and neg words in this claim (needed to convert to line.column format later)
        return words_sorted[max(0,len(words_sorted)-10):],words_sorted[:min(10,len(words_sorted))]
      
    # get additional features from word level  
    def get_addData(self,text):
            # 1.get pos and neg words from previously processed data
        negwords=[]
        for line in open('/Volumes/Zhipeng/patent_dataset/negword_freq.csv'):
            parts=line.lower().split(',')
            negwords.append(parts[0])
        
        poswords=[]
        for line in open('/Volumes/Zhipeng/patent_dataset/posword_freq.csv'):  
            parts=line.lower().split(',')
            poswords.append(parts[0])
            
            # 2.build dictionary for count vectorize
        negword_ids = {}
        negrev_word_ids = {}
        for i, x in enumerate(negwords):
            negword_ids[x] = i + 1  # so we can pad with 0s
            negrev_word_ids[i + 1] = x   
        
        posword_ids = {}
        posrev_word_ids = {}
        for i, x in enumerate(poswords):
            posword_ids[x] = i + 1  # so we can pad with 0s
            posrev_word_ids[i + 1] = x
    
            # 3.get additional features
        neg_vector=self.count_vectorize(negword_ids,text,len(negwords))
        pos_vector=self.count_vectorize(posword_ids,text,len(poswords))
        return neg_vector+pos_vector
    
    # get main features from sentence level 
    def get_mainData(self,text):
        self.rev_word_ids = np.load('/Volumes/Zhipeng/patent_dataset/revdic.npy').item()
        word_ids = np.load('/Volumes/Zhipeng/patent_dataset/worddic.npy').item()
        t_ids = [word_ids[re.sub('[^a-zA-Z]+', '', x)] if re.sub('[^a-zA-Z]+', '', x) in word_ids else 0 for x in text.split()]
        item=[0]*maxlen
        item[:min(maxlen,len(t_ids))]=t_ids[:min(maxlen,len(t_ids))]
        return item
        
    
    def load_model(self):
        from keras.models import Sequential
        from keras.models import model_from_json
        # load json and create model
        json_file = open('/Users/mac/Machine-Learning-NLP/new_model/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        
        # load weights into new model
        self.model.load_weights("/Users/mac/Machine-Learning-NLP/new_model/model.h5")
        self.model_status['text'] = "Model Loaded from disk"
        
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])       

        # load json and create model
        json_file = open('/Users/mac/Machine-Learning-NLP/new_model/emb_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.emb_model = model_from_json(loaded_model_json)
        
        # load weights into new model
        self.emb_model.load_weights("/Users/mac/Machine-Learning-NLP/new_model/emb_model.h5")
        self.model_status['text'] = "Model Loaded and Compiled"
    def init_gui(self):
        """Builds GUI."""
        self.root.title('Patent Claim Scoring System')
        self.root.option_add('*tearOff', 'FALSE')

        self.grid(column=0, row=0, sticky='nsew')

        self.menubar = tkinter.Menu(self.root)

        self.menu_file = tkinter.Menu(self.menubar)
        self.menu_file.add_command(label='Exit', command=self.on_quit)

        self.menu_edit = tkinter.Menu(self.menubar)

        self.menubar.add_cascade(menu=self.menu_file, label='File')
        self.menubar.add_cascade(menu=self.menu_edit, label='Edit')

        self.root.config(menu=self.menubar)

        
        # Labels that remain constant throughout execution.
        ttk.Label(self, text='Please Type Your Idea').grid(column=0, row=0,
                columnspan=4, rowspan=2)
        
        
        ttk.Separator(self, orient='horizontal').grid(column=0,
                row=2, columnspan=15, sticky='ew')
        
        # Add initialize button
        self.minit_button = ttk.Button(self, text='Initialize the model',
                command=self.load_model)
        self.minit_button.grid(row=0, column=5, columnspan=4,rowspan=2)
        
        # Add label to show status
        self.model_status = ttk.Label(self, text='Uninitialized')
        self.model_status.grid(row=1, column=10)
        
        # Add text box to collect input
        self.textbox = Text(self, height=20, width=60)
        self.textbox.grid(row=3,column=0, columnspan=4)
        
        # Add Test button
        self.calc_button = ttk.Button(self, text='Test',
                command=self.calculate)
        self.calc_button.grid(row=4, columnspan=4)
        
         # Add label to show score
        self.answer_frame = ttk.LabelFrame(self, text='Score',
                height=100)
        self.answer_frame.grid(column=0, row=5, columnspan=4, sticky='nesw')

        self.answer_label = ttk.Label(self.answer_frame, text='')
        self.answer_label.grid(column=5, row=5)
        
        ttk.Separator(self, orient='horizontal').grid(column=0,
                row=5, columnspan=9, sticky='ew')
        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)

if __name__ == '__main__':
    root = tkinter.Tk()
#    root.geometry("800x800")
    Adder(root)
    root.mainloop()