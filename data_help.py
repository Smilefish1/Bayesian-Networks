# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:27:46 2019

@author: cyq
"""
import codecs
import random
import numpy as np
from tflearn.data_utils import pad_sequences
from collections import Counter
import pickle
import re
import os
#import h5py
PAD_ID = 0
UNK_ID=1
_PAD="_PAD"
_UNK="UNK"

def transform_multilabel_as_multihot(label_list,label_size):
    """
    convert to multi-hot style
    :param label_list: e.g.[0,1,4], here 4 means in the 4th position it is true value(as indicate by'1')
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result=np.zeros(label_size)
    #set those location as 1, all else place as 0.
    result[label_list] = 1
    return result


def load_data_multilabel(traning_data_path,vocab_word2index, label2index,sentence_len,training_portion=0.95):
    """
    convert data as indexes using word2index dicts.
    :param traning_data_path:
    :param vocab_word2index:
    :param vocab_label2index:
    :return:
    """
    file_object = codecs.open(traning_data_path, mode='r', encoding='utf-8')
    lines = file_object.readlines()
    random.shuffle(lines)
    label_size=len(label2index)
    X = []
    Y = []
    for i,line in enumerate(lines):
        raw_list=re.split('__label__| |\n', line)
        input_list =[i for i in raw_list[5:-1] if i != '']
        x=[vocab_word2index.get(x,UNK_ID) for x in input_list]
        label_list = (raw_list[1]+ ',' + raw_list[3]).split(',')
        #print(label_list)
        #label_list=[l.strip().replace(" ", "") for l in label_list if l != '']
        label_list=[label2index[label] for label in label_list]
        y=transform_multilabel_as_multihot(label_list,label_size)
        X.append(x)
        Y.append(y)
        if i<10:
            print(i,"line:",line)

    X = pad_sequences(X, maxlen=sentence_len, value=0.)  # padding to max length
    number_examples = len(lines)
    training_number=int(training_portion* number_examples)
    train = (X[0:training_number], Y[0:training_number])
    valid_number=min(1000,number_examples-training_number)
    test = (X[training_number+ 1:training_number+valid_number+1], Y[training_number + 1:training_number+valid_number+1])
    return train,test

#test
# =============================================================================
# with open('word_index', 'rb') as f:
#     label2index,vocab_index2word,vocab_word2index = pickle.load(f)
# train,test=load_data_multilabel('adjust_label_train_corpus_completion',vocab_word2index, label2index,25,training_portion=0.95)
# 
# =============================================================================



#use pretrained word embedding to get word vocabulary and labels, and its relationship with index
def create_vocabulary(training_data_path,vocab_size,name_scope='cnn'):
    """
    create vocabulary
    :param training_data_path:
    :param vocab_size:
    :param name_scope:
    :return:
    """

    cache_vocabulary_label_pik='cache'+"_"+name_scope # path to save cache
    if not os.path.isdir(cache_vocabulary_label_pik): # create folder if not exists.
        os.makedirs(cache_vocabulary_label_pik)

    # if cache exists. load it; otherwise create it.
    cache_path =cache_vocabulary_label_pik+"/"+'vocab_label.pik'
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            return pickle.load(data_f)
    else:
        vocabulary_word2index={}
        vocabulary_index2word={}
        vocabulary_word2index[_PAD]=PAD_ID
        vocabulary_index2word[PAD_ID]=_PAD
        vocabulary_word2index[_UNK]=UNK_ID
        vocabulary_index2word[UNK_ID]=_UNK

        vocabulary_label2index={}
        vocabulary_index2label={}

        #1.load raw data
        file_object = codecs.open(training_data_path, mode='r', encoding='utf-8')
        lines=file_object.readlines()
        #2.loop each line,put to counter
        c_inputs= Counter()
        c_labels=Counter()
        input_list=[]
        label_list=[]
        for line in lines[:]:
            raw_list=re.split('__label__| |\n', line)
            input_list =[i for i in raw_list[5:-1] if i != '']
            label_list = [raw_list[1] + raw_list[3]]
            c_inputs.update(input_list)
            c_labels.update(label_list)

        #return most frequency words
        vocab_list=c_inputs.most_common(vocab_size)
        label_list=c_labels.most_common()
        #put those words to dict
        for i,tuplee in enumerate(vocab_list):
            word,_=tuplee
            vocabulary_word2index[word]=i+2
            vocabulary_index2word[i+2]=word

        for i,tuplee in enumerate(label_list):
            label,_=tuplee;label=str(label)
            vocabulary_label2index[label]=i
            vocabulary_index2label[i]=label

        #save to file system if vocabulary of words not exists.
#        if not os.path.exists(cache_path):
#            with open(cache_path, 'ab') as data_f:
#                pickle.dump((vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label), data_f)
    return vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label

# =============================================================================
# test
#vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label=create_vocabulary('adjust_label_train_corpus_completion',10000,name_scope='cnn')
# 
# =============================================================================



# =============================================================================
# 
# 导入词向量模型
# from gensim.models import KeyedVectors
# general_embedding_dir = 'GoogleNews-vectors-negative300.bin'
# model =KeyedVectors.load_word2vec_format(general_embedding_dir, limit=500000, binary=True)
# print(model.get_vector('type'))
# 
# =============================================================================

           
def get_vec(count,model,cache_path):
    vocab_index2vec={}
    for i in range(len(count)):
        try:
            vocab_index2vec[i]=model.get_vector(count[i])
        except:
            vocab_index2vec[i]=np.random.uniform(-1,1,300)
#    if not os.path.exists(cache_path):
#            with open(cache_path, 'ab') as data_f:
#                pickle.dump((vocab_index2vec), data_f)
    return vocab_index2vec

#vocab_index2vec=get_vec(vocabulary_index2word,model,'vector2index')          
import h5py      
def load_data(cache_file_h5py,cache_file_pickle):
    """
    load data from h5py and pickle cache files, which is generate by take step by step of pre-processing.ipynb
    :param cache_file_h5py:
    :param cache_file_pickle:
    :return:
    """
    if not os.path.exists(cache_file_h5py) or not os.path.exists(cache_file_pickle):
        raise RuntimeError("############################ERROR##############################\n. "
                           "please download cache file, it include training data and vocabulary & labels. "
                           "link can be found in README.md\n download zip file, unzip it, then put cache files as FLAGS."
                           "cache_file_h5py and FLAGS.cache_file_pickle suggested location.")
    print("INFO. cache file exists. going to load cache file")
    f_data = h5py.File(cache_file_h5py, 'r')
    print("f_data.keys:",list(f_data.keys()))
    train_X=f_data['train_x'] # np.array(
    print("train_X.shape:",train_X.shape)
    train_Y=f_data['train_y'] # np.array(
    print("train_Y.shape:",train_Y.shape,";")
    test_X=f_data['test_x'] # np.array(
    test_Y=f_data['test_y'] # np.array(
    #print(train_X)
    #f_data.close()

    word2index, label2index=None,None
    with open(cache_file_pickle, 'rb') as data_f_pickle:
        word2index,index2word,label2index=pickle.load(data_f_pickle)
    print("INFO. cache file load successful...")
    return word2index,index2word, label2index,train_X,train_Y,test_X,test_Y

#import pickle
#
#word2index, index2word,label2index, trainX, trainY, testX, testY=load_data('new_data', 'word_label_index')

def get_target_label_set(eval_y):#函数的功能是给出label的位置
    eval_y_short=[] #will be like:[22,642,1391]
    for index,label in enumerate(eval_y):
        if label>0:
            eval_y_short.append(index)
    return eval_y_short


 

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    