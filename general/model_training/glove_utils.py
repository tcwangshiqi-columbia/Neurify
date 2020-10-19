import numpy as np
from collections import defaultdict
import math
import pickle
from numpy import linalg as LA
import time
from embedding import Embedding

def pad_sequences(X, maxlen, emb_size=50):
    for i in range(len(X)):
        if len(X[i]) > maxlen:
            X[i] = X[i][:maxlen]
        elif len(X[i]) < maxlen:
            pad = np.zeros(shape=(maxlen-len(X[i]), emb_size))
            X[i] = np.append(X[i], pad, axis=0)
    return X


def index_to_word(word2index) :
    index2word = {value:key for key,value in word2index.items()}
    index2word[0] = '<PAD>'
    index2word[1] = '<START>'
    index2word[2] = '<UNK>'
    return index2word


# Based on https://github.com/guillaume-chevalier/GloVe-as-a-TensorFlow-Embedding-Layer/blob/master/GloVe-as-TensorFlow-Embedding-Tutorial.ipynb
def load_embedding(glove):
    '''
        Load word embeddings from file.
    '''
    word_to_index_dict = dict()
    index_to_embedding_array = []
    
    with open(glove, 'r', encoding="utf-8") as glove_file:
        for (i, line) in enumerate(glove_file):
            split = line.split(' ')            
            word = split[0]            
            representation = split[1:]
            representation = np.array(
                [float(val) for val in representation]
            )
            # use +3 because actual word indexes start at 3 while indexes 0,1,2 are for
            # <PAD>, <START>, and <UNK>
            word_to_index_dict[word] = i+3
            index_to_embedding_array.append(representation)

    _WORD_NOT_FOUND = [0.0]* len(representation)  # Empty representation for unknown words.
    _PAD = 0
    _START = 1
    _UNK = 2
    word_to_index_dict['<PAD>'] = 0
    word_to_index_dict['<UNK>'] = 2
    word_to_index_dict = defaultdict(lambda: _UNK, word_to_index_dict)
    index_to_word_dict = index_to_word(word_to_index_dict)
    # three 0 vectors for <PAD>, <START> and <UNK>
    index_to_embedding_array = np.array(3*[_WORD_NOT_FOUND] + index_to_embedding_array )
    return word_to_index_dict, index_to_word_dict, index_to_embedding_array


def load_binary_embedding(words):
    num_digits = math.ceil(math.log2((1+len(words))))
    word_to_index_dict = dict()
    index_to_embedding_array = []
    for i,w in zip(range(len(words)), words):
        representation = [float(digit) for digit in bin(i+1)[2:]][::-1]
        representation.extend([0. for _ in range(num_digits-len(representation))])
        index_to_embedding_array.append(representation[::-1])
        word_to_index_dict[w] = i+3
    _WORD_NOT_FOUND = [0.0]* num_digits  # Empty representation for unknown words.
    _PAD = 0
    _START = 1
    _UNK = 2
    word_to_index_dict['<PAD>'] = 0
    word_to_index_dict['<UNK>'] = 2
    word_to_index_dict = defaultdict(lambda: _UNK, word_to_index_dict)
    index_to_word_dict = index_to_word(word_to_index_dict)
    index_to_embedding_array = np.array(3*[_WORD_NOT_FOUND] + index_to_embedding_array )
    return word_to_index_dict, index_to_word_dict, index_to_embedding_array


def load_syn_dict(filename = 'data/syn_dict/syn_dict_glove300.pickle', N = 10):
    '''
        Load cached synonyms dictionary.
    '''
    try:
        file = open(filename, 'rb')
        syn_dict = pickle.load(file)
        syn_dict = {word: neighbors[:N] for word, neighbors in syn_dict.items()}
        return syn_dict
    except:
        print("ERROR: Could not load synonyms dictionary.")
    return dict()


def load_dist_dict(filename = 'data/syn_dict/dist_dict_glove300.pickle', N = 10):
    '''
        Load cached dictionary with distances to nearest neighbors.
    '''
    try :
        file = open(filename, 'rb')
        dist_dict = pickle.load(file)
        dist_dict = {word: distances[:N] for word, distances in dist_dict.items()}
        return dist_dict
    except:
        print("ERROR: Could not load distances to nearest neighbors dictionary")
    return dict()
