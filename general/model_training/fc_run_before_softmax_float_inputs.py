'''Directly running this python script will train the
fully connected DNN for MNIST dataset.
This script can also be loaded as the pretrained model
by others
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, Dense, Dropout, Conv1D, Conv2D, Flatten, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, BatchNormalization  #create model
import numpy as np
import sys

from glove_utils import load_embedding, pad_sequences
from text_utils import clean_text, stem_lem

def FCModel(weights_path):
    maxlen = 10
    emb_dims = 5
    model = Sequential()
    model.add(Dense(16, input_shape=(maxlen*emb_dims,), activation='relu'))
    #model.add(Dense(32, activation='relu'))
    model.add(Dense(2,name='before_softmax'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'])
    model.load_weights(weights_path)
    return model

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Need weight path.')
        sys.exit(0)

    model = FCModel(sys.argv[1])

    input_without_padding = '... spellbinding fun and deliciously exploitative'
    num_words = 10
    emb_dims = 5
    model_input_shape = (1, emb_dims*num_words)

    path_to_embeddings = './embeddings/'
    EMBEDDING_FILENAME = path_to_embeddings+'custom-embedding-SST.{}d.txt'.format(emb_dims)
    word2index, _, index2embedding = load_embedding(EMBEDDING_FILENAME)
    embedding = lambda W : np.array([index2embedding[word2index[w]] for w in W]).reshape(model_input_shape)

    input_without_padding = input_without_padding.lower().split(' ') 
    input_ = input_without_padding[:num_words] + ['<PAD>']*(num_words - len(input_without_padding))
    x = embedding(input_)
    with open('../text_inputs/new_mara_in','r') as f:
    #with open('../text_inputs/test','r') as f:
        text_in = f.read().split(',')[:-1]
    text_in = list(map(float,text_in))
    text_in = np.array([text_in])
   # print(text_in)
   # print(x)
    x = text_in
    x = (x+1)/2


    extractor = tf.keras.models.Model(inputs=model.inputs,outputs=model.layers[-2].output)
    features = extractor.predict(x)
    print(features)
