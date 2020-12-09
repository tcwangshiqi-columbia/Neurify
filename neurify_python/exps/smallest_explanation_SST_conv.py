import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# quick fix to specify the GPUs to use, comment out if there is no one using 100% of them
os.environ["CUDA_VISIBLE_DEVICES"]="4,5"

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
from glove_utils import load_embedding, pad_sequences
from text_utils import clean_text, stem_lem

import json
import math

def to_words(h, inp_w_pad, wsize, emb_dim):
    word_starts = h[0::wsize]
    words = []
    for ws in word_starts:
        idx = int(ws/emb_dim)
        words.append(inp_w_pad[idx])
    return words
    

# Parse epsilon and window size from command line
parser = argparse.ArgumentParser(description='Input string, window size and epsilon can be passed as arguments.')
parser.add_argument('-i', '--input', dest='input', type=str, default='... spellbinding fun and deliciously exploitative', help='input string')
parser.add_argument('-w', '--window-size', dest='wsize', type=int, default=5, help='window size')
parser.add_argument('-e', '--eps', dest='eps', type=float, default=0.05, help='epsilon for input bounds')
parser.add_argument('-c', '--config', dest='config', type=str, default='config.json',help='config file with network spec')
args = parser.parse_args()
# Assign eps and wsize
window_size = args.wsize
eps = args.eps
input_without_padding = args.input
with open(args.config) as f:
    config = json.load(f)

print('Starting experiment with eps = '+ str(eps) + ', on review: "'+input_without_padding+'"')

# Global Variables
verbose = config['verbose']
randomize_pickfalselits = config['random_pfl']  # shuffle free variables before feeding them to pickfalselits
HS_max_len = config['hs_maxlen']  # max size of GAMMA in smallest_explanation

# Load model and test the input_ review
model = load_model(config['keras_file'])
emb_dims = config['emb_dims']
in_shape = model.layers[0].input.shape
num_words = int(math.ceil((in_shape[1]*in_shape[2])/emb_dims))
model_input_shape = (1, in_shape[1], in_shape[2], in_shape[3])
input_len = emb_dims*num_words

# Load embedding
word2index, _, index2embedding = load_embedding(config['embedding_file'])
embedding = lambda W : np.array([index2embedding[word2index[w]] for w in W])

# Review + <pad>(s)
input_without_padding = input_without_padding.lower().split(' ') 
input_ = input_without_padding[:num_words] + ['<PAD>']*(num_words - len(input_without_padding))
x = embedding(input_)
sq = in_shape[1]
x = x.flatten()[:(sq**2)].reshape(sq,sq)
# normalise
x = (x+1)/2
x = x.reshape(*model_input_shape)

prediction = model.predict(x)
y_hat = np.argmax(prediction)
c_hat = np.max(prediction)

output_constraints = [y_hat, (1 if y_hat==0 else 0), -1e-3]

# Start Main
# Args and procedures for parallelizing search of an attack
# args are in order: model, input_, y_hat, num_classes, targeted=True,
#                    loss="categorical_cross_entropy", mask=[], 
#                    eps=1e-3, epochs=100, return_dictionary=False

net_path = config['nnet_file']

h, exec_time, GAMMA = smallest_explanation(model, net_path, x, eps, y_hat, output_constraints, window_size, 
                                           adv_attacks=False, adv_args=None, sims=0, randomize_pickfalselits=randomize_pickfalselits, HS_maxlen=HS_max_len, verbose=False)

word_exp = to_words(h,input_,window_size,emb_dims)

# Report ORE found
out_string = "Input: " + str(input_) + "\n"
out_string += "Eps: " + str(eps) + " Window size: "+str(window_size) + "\n"
out_string += "Classification: {} (confidence {})".format(y_hat, c_hat) + "\n"
out_string += "MSE found: {} (size {})".format(h, int(len(h)/window_size)) + "\n"
out_string += "MSE words: " + str(word_exp) + "\n"
out_string += "Complementary set of MSE: {}".format([i for i in range(input_len) if i not in h]) + "\n"
out_string += "Execution Time (s): {}".format(exec_time) + "\n\n"
print(out_string,end='')

out_file = config['nnet_file'].split('/')[-1][:-5] + "_results.txt"
out_file_path = os.path.join(config['out_folder'],out_file)
with open(out_file_path,'a') as f:
    f.write(out_string)
