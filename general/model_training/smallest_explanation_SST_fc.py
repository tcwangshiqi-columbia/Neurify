"""
Take a model saved with Keras+Tensorflow and save the frozen graph in a .pb format.model.layers[idx].get_config()
Execute this with the command %run ./examples/minimal_subset_SST.py
"""
import argparse
import copy as cp
import numpy as np
import sys
import time
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import load_model, Sequential
from keras.layers import Dense, Input, Activation
from keras.models import Model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, Dense, Dropout, Conv1D, Conv2D, Flatten, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, BatchNormalization  #create model

sys.path.append('./../../../expl/Marabou')
# import Marabou
from maraboupy import Marabou
from maraboupy import MarabouUtils, MarabouCore
# import abduction algorithms
from abduction_algorithms import logger, freeze_session, smallest_explanation 
# import embeddings relative path
from glove_utils import load_embedding, pad_sequences
from text_utils import clean_text, stem_lem

def FCModel(weights_path):
    maxlen = 10
    emb_dims = 5
    model = Sequential()
    model.add(Dense(32, input_shape=(maxlen*emb_dims,), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2,name='before_softmax'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'])
    model.load_weights(weights_path)
    return model

# Parse epsilon and window size from command line
parser = argparse.ArgumentParser(description='Input string, window size and epsilon can be passed as arguments.')
parser.add_argument('-i', '--input', dest='input', type=str, default='You really have to salute writer-director Haneke ( he adapted Elfriede Jelinek novel )', help='input string')
parser.add_argument('-w', '--window-size', dest='wsize', type=int, default=5, help='window size')
parser.add_argument('-e', '--epsilon', dest='eps', type=float, default=0.05, help='epsilon for input bounds')
args = parser.parse_args()
# Assign eps and wsize
window_size = args.wsize
eps = args.eps
input_without_padding = '... spellbinding fun and deliciously exploitative'

# Global Variables
verbose = True
randomize_pickfalselits = False  # shuffle free variables before feeding them to pickfalselits
frozen_graph_path = 'tf_model.pb'
frozen_graph_prefix = './'
emb_dims = 5
input_len = emb_dims*10
num_words = int(input_len/emb_dims)
model_input_shape = (1, emb_dims*num_words)
HS_maxlen = 100000000  # max size of GAMMA in smallest_explanation

# Load model and test the input_ review
model = FCModel('models/SST_fc_5d_10inp_format_32_16_2_acc_81.6.h5')

# Load embedding
path_to_embeddings = './embeddings/'
EMBEDDING_FILENAME = path_to_embeddings+'custom-embedding-SST.{}d.txt'.format(emb_dims)
word2index, _, index2embedding = load_embedding(EMBEDDING_FILENAME)
embedding = lambda W : np.array([index2embedding[word2index[w]] for w in W]).reshape(model_input_shape)

# Review + <pad>(s)
input_without_padding = input_without_padding.lower().split(' ') 
input_ = input_without_padding[:num_words] + ['<PAD>']*(num_words - len(input_without_padding))
x = embedding(input_)

input_shape = x.shape
prediction = model.predict(x)
input_ = x.flatten().tolist()
y_hat = np.argmax(prediction)
c_hat = np.max(prediction)
logger("Classifiation for the input is {} (confidence {})".format(y_hat, c_hat), verbose)

# Graph
frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
# Remove softmax layer
del frozen_graph.node[-1] 
tf.train.write_graph(frozen_graph, frozen_graph_prefix, frozen_graph_path, as_text=False)
filename = frozen_graph_prefix + frozen_graph_path
output_constraints = [y_hat, (1 if y_hat==0 else 0), -1e-3]

# Start Main
# Args and procedures for parallelizing search of an attack
# args are in order: model, input_, y_hat, num_classes, targeted=True,
#                    loss="categorical_cross_entropy", mask=[], 
#                    eps=1e-3, epochs=100, return_dictionary=False

sims = 10  # arguments to run sparseRS routine (minimzation of #attacks)
h, exec_time, GAMMA = smallest_explanation(model, filename, x, eps, y_hat, output_constraints, window_size, 
                                           adv_attacks=False, adv_args=None, sims=sims, randomize_pickfalselits=randomize_pickfalselits, HS_maxlen=HS_maxlen, verbose=verbose)

# Report MSR found
logger("Minimum Size Explanation found {} (size {})".format(h, len(h)/window_size), True)
logger("Complementary set of Minimum Size Explanation is {}".format([i for i in range(input_len) if i not in h]), True)
logger("Execution Time: {}".format(exec_time), True)
