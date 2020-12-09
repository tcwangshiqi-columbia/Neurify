'''Directly running this python script will train the
fully connected DNN for MNIST dataset.
This script can also be loaded as the pretrained model
by others
'''
import os
import sys

# quick fix to specify the gpus to use, comment out if there is no one using 100% of them
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Embedding, Dense, Dropout, Conv1D, Conv2D, Flatten, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, BatchNormalization  #create model
import numpy as np
from glove_utils import load_embedding, pad_sequences
from text_utils import clean_text, stem_lem

import math

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Need model path and embedding dim.')
        sys.exit(0)

    model = load_model(sys.argv[1])
    emb_dims = int(sys.argv[2])

    input_without_padding = '... spellbinding fun and deliciously exploitative'
    in_shape = model.layers[0].input.shape
    num_words = int(math.ceil((in_shape[1]*in_shape[2])/emb_dims))
    model_input_shape = (1, in_shape[1], in_shape[2], in_shape[3])

    path_to_embeddings = './embeddings/'
    EMBEDDING_FILENAME = path_to_embeddings+'custom-embedding-SST.{}d.txt'.format(emb_dims)
    word2index, _, index2embedding = load_embedding(EMBEDDING_FILENAME)
    embedding = lambda W : np.array([index2embedding[word2index[w]] for w in W])

    input_without_padding = input_without_padding.lower().split(' ') 
    input_ = input_without_padding[:num_words] + ['<PAD>']*(num_words - len(input_without_padding))
    x = embedding(input_)
    sq = in_shape[1]
    x = x.flatten()[:(sq**2)].reshape(sq,sq)
    #with open('../text_inputs/1_spellbinding.csv','r') as f:
    #with open('../text_inputs/test','r') as f:
    #    text_in = f.read().split(',')[:-1]
   # text_in = list(map(float,text_in))
   # text_in = np.array([text_in])
   # print(text_in)
   # print(x)
   # x = text_in
    x = (x+1)/2
    print(','.join(list(map(str,x.flatten()))))
    x = x.reshape(*model_input_shape)

    print(model.layers)

    #print(model.layers[0].weights)
    print()

    
    '''
    for ln in range(len(model.layers)):
        extractor = tf.keras.models.Model(inputs=model.inputs,outputs=model.layers[ln].output)
        features = extractor.predict(x)
        print("output", features)
        print('layer',ln)
        print('layer input shape',model.layers[ln].input.shape)
        if not ln == 1:
            print('layer weights shape',model.layers[ln].get_weights()[0].shape)
        print('output shape',features.shape)
        print()
    '''

    extractor = tf.keras.models.Model(inputs=model.inputs,outputs=model.layers[-2].output)
    features = extractor.predict(x)
    print(features)
