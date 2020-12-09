import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# quick fix to specify the GPUs to use, comment out if there is no one using 100% of them
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import argparse
import copy as cp
import numpy as np
import sys
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Embedding, Dense, Dropout, Conv1D, Conv2D, Flatten, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, BatchNormalization  #create model

# add our python source files to our path
sys.path.append('../src/')
# import abduction algorithms
from abduction_algorithms import logger, smallest_explanation 
# import embeddings relative path
import entails
from entails import entails
from glove_utils import load_embedding, pad_sequences
from text_utils import clean_text, stem_lem

import json
import itertools
import ast

def get_all_of_length(orig_input,mse_len):
    if mse_len == 0:
        return []
    return_list = itertools.combinations(zip(orig_input,list(range(len(orig_input)))),mse_len)
    return_list = [list(x) for x in return_list]
    return return_list

def convert_to_numbers(word_list,window_size):
    numbers = []
    for w in word_list:
        numbers += list(range(w[1]*window_size,(w[1]*window_size)+window_size))
    return numbers

def read_results(filename):
    storage_dict = dict()
    with open(filename) as f:
        txt = f.read()

    txt_blocks = txt.split('\n\n')[:-1]

    for block in txt_blocks:
        lines = block.split('\n')
        inputs = ast.literal_eval(' '.join(lines[0].split(' ')[1:]))
        eps = float(lines[1].split(' ')[1])
        classification = int(lines[2].split(' ')[1])
        confidence = float(lines[2].split(' ')[3][:-1])
        mse = ast.literal_eval(' '.join(lines[3].split(' ')[2:]).split('(')[0][:-1])
        mse_words = ast.literal_eval(' '.join(lines[4].split(' ')[2:]))
        exec_time = float(lines[6].split(' ')[3])

        window_size = 5

        # input + eps is key so we can match up
        key = str([inputs,eps])
        storage_dict[key] = dict()
        storage_dict[key]['inputs'] = inputs
        storage_dict[key]['eps'] = eps
        storage_dict[key]['window_size'] = window_size
        storage_dict[key]['mse'] = mse
        storage_dict[key]['mse_words'] = mse_words
        storage_dict[key]['exec_time'] = exec_time

    return storage_dict

def get_min_expl(min_expl,inputs):
    min_expl_arr = []
    c = 0
    while c < len(min_expl):
        start = int(min_expl[c]/5)
        if start < len(inputs):
            word = inputs[start]
            min_expl_arr.append(word)
            c += 5
        else:
            min_expl_arr.append('<PAD>')
            c += 5
    return min_expl_arr


window_size = 5
emb_dims = 5
input_len = emb_dims*10
num_words = int(input_len/emb_dims)
model_input_shape = (1, emb_dims*num_words)

if not len(sys.argv) == 3:
    print('Need network config file and results file.')
    sys.exit(0)

with open(sys.argv[1]) as f:
    config = json.load(f)

model = load_model(config['keras_file'])

word2index, _, index2embedding = load_embedding(config['embedding_file'])
embedding = lambda W : np.array([index2embedding[word2index[w]] for w in W]).reshape(model_input_shape)

net_path = config['nnet_file']

# start processing results
results_dict = read_results(sys.argv[2])
for key in list(results_dict.keys()):
    pair = ast.literal_eval(key)
    in_list = pair[0]
    eps = float(pair[1])

    min_expl = results_dict[key]['mse']
    if len(min_expl) == 0:
        continue

    input_without_padding = in_list
    input_ = input_without_padding[:num_words] + ['<PAD>']*(num_words - len(input_without_padding))
    orig_input = input_.copy()
    x = embedding(input_)
    x = (x+1)/2

    input_shape = x.shape
    prediction = model.predict(x)
    input_ = x.flatten().tolist()
    y_hat = np.argmax(prediction)
    c_hat = np.max(prediction)

    res_orig = results_dict[key]['mse_words']
    all_of_len = get_all_of_length(orig_input,len(res_orig))

    actual_expls = []
    actual_words = []
    actual_all = []
    for smallest_expl in all_of_len:
        hit_set = convert_to_numbers(smallest_expl,5)
        words = [w[0] for w in smallest_expl]
        flat_x = x.flatten()
        res = 1
        try:
            res = entails(hit_set,len(hit_set),flat_x, len(flat_x),eps,net_path)
        except:
            pass
        if res == 0:
            actual_expls.append(hit_set)
            actual_words.append(words)
            actual_all.append(smallest_expl)

    out_string = "Input: "+ str(orig_input) + "\n"
    out_string += "Eps: "+ str(eps) + "\n"
    out_string += "Original smallest expl: " + str(res_orig) + "\n"
    out_string += "Number smallest expls: "+str(len(actual_all)) +"\n"
    out_string += "In form: [(word, index_in_input)]\n"
    out_string += "\n"
    for ac in actual_all:
        out_string += str(ac) + "\n"
    out_string += "\n"
    out_string += '=====================================================================\n'
    print(out_string)
    output_file = 'all_smallest_convtest.txt'
    with open(output_file,'w') as f:
        f.write(out_string)
