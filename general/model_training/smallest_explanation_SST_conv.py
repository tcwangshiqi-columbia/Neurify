"""
Take a model saved with Keras+Tensorflow and save the frozen graph in a .pb format.model.layers[idx].get_config()
Execute this with the command %run ./examples/minimal_subset_SST.py
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import argparse
import copy as cp
import numpy as np
import sys
import time
import tensorflow as tf
from tensorflow import keras
print(tf.version.VERSION)
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, Dense, Dropout, Conv1D, Conv2D, Flatten, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, BatchNormalization  #create model

# import Marabou
from maraboupy import Marabou
from maraboupy import MarabouUtils, MarabouCore
# import abduction algorithms
from abduction_algorithms import logger, smallest_explanation 
# import embeddings relative path
from glove_utils import load_embedding, pad_sequences
from text_utils import clean_text, stem_lem

def ConvModel(weights_path):
    maxlen = 9
    ksize = int(maxlen**0.5)
    kernel_dim = 2
    padding = "valid"
    stride = (1,1)
    emb_dims = 5
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=kernel_dim, strides=stride, padding=padding,
                     input_shape=(ksize, ksize, emb_dims), activation='relu'))
    model.add(Flatten())
    model.add(Dense(2,name='before_softmax'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'])
    model.load_weights(weights_path)
    return model

maxlen = 9
ksize = int(maxlen**0.5)
kernel_dim = 2
padding = "valid"
stride = (1,1)
emb_dims = 5
# Parse epsilon and window size from command line
parser = argparse.ArgumentParser(description='Input string, window size and epsilon can be passed as arguments.')
parser.add_argument('-i', '--input', dest='input', type=str, default='You really have to salute writer-director Haneke ( he adapted Elfriede Jelinek novel )', help='input string')
parser.add_argument('-w', '--window-size', dest='wsize', type=int, default=5, help='window size')
parser.add_argument('-e', '--epsilon', dest='eps', type=float, default=0.05, help='epsilon for input bounds')
args = parser.parse_args()
# Assign eps and wsize
window_size = args.wsize
eps = args.eps
input_without_padding = args.input

print(eps, input_without_padding)

# Global Variables
verbose = True
randomize_pickfalselits = False  # shuffle free variables before feeding them to pickfalselits
emb_dims = 5
input_len = emb_dims*9
num_words = int(input_len/emb_dims)
model_input_shape = (1, emb_dims*num_words)
HS_maxlen = 100000  # max size of GAMMA in smallest_explanation

# Load model and test the input_ review
model = ConvModel('models/SST_conv_5d_9inp_norm.h5')

# Load embedding
path_to_embeddings = './embeddings/'
EMBEDDING_FILENAME = path_to_embeddings+'custom-embedding-SST.{}d.txt'.format(emb_dims)
word2index, _, index2embedding = load_embedding(EMBEDDING_FILENAME)
embedding = lambda W : np.array([index2embedding[word2index[w]] for w in W]).reshape(model_input_shape)

# Review + <pad>(s)
input_without_padding = input_without_padding.lower().split(' ') 
input_ = input_without_padding[:num_words] + ['<PAD>']*(num_words - len(input_without_padding))
x = embedding(input_)
# normalise
x = (x+1)/2
print(x)
x = x.reshape(1,ksize, ksize, emb_dims)

input_shape = x.shape
prediction = model.predict(x)
print(prediction)
input_ = x.flatten().tolist()
y_hat = np.argmax(prediction)
c_hat = np.max(prediction)
print("Classifiation for the input is {} (confidence {})".format(y_hat, c_hat))

model_before_softmax = tf.keras.models.Model(inputs=model.inputs,outputs=model.layers[-2].output)
filename = 'models/SST_conv_5d_9inp_norm_f/' 
tf.saved_model.save(model_before_softmax,filename)

output_constraints = [y_hat, (1 if y_hat==0 else 0), -1e-3]

# Start Main
# Args and procedures for parallelizing search of an attack
# args are in order: model, input_, y_hat, num_classes, targeted=True,
#                    loss="categorical_cross_entropy", mask=[], 
#                    eps=1e-3, epochs=100, return_dictionary=False
'''
with open('../text_inputs/spellbindingfunanddeliciouslyexploitative_adv','r') as f:
    text_in = f.read().split(',')[:-1]
text_in = list(map(float,text_in))
text_in = np.array([text_in])
network = Marabou.read_tf(filename)
print('-----------')
for n in range(len(text_in[0])):
    equation = MarabouUtils.Equation()
    equation.addAddend(1, n)
    equation.setScalar(text_in[0][n])
    network.addEquation(equation)

vals,b = network.solve(verbose=0)
print(vals)
print('------')
'''

sims = 10  # arguments to run sparseRS routine (minimzation of #attacks)
h, exec_time, GAMMA = smallest_explanation(model, filename, x, eps, y_hat, output_constraints, window_size, 
                                           adv_attacks=False, adv_args=None, sims=sims, randomize_pickfalselits=randomize_pickfalselits, HS_maxlen=HS_maxlen, verbose=False)

# Report MSR found
print()
print("Minimum Size Explanation found {} (size {})".format(h, len(h)/window_size))
print("Complementary set of Minimum Size Explanation is {}".format([i for i in range(input_len) if i not in h]))
print("Execution Time: {}".format(exec_time))
