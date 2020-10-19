'''Directly running this python script will train the
fully connected DNN for MNIST dataset.
This script can also be loaded as the pretrained model
by others
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.layers import Dense, Input, Activation
from keras.models import Model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, Dense, Dropout, Conv1D, Conv2D, Flatten, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, BatchNormalization  #create model
import numpy as np
import sys

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

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Need weight path.')
        sys.exit(0)

    model = FCModel(sys.argv[1])

    layer_num = len(model.layers)-1 # before softmax layer
    max_neurons = 0
    weights = []
    bias = []
    for ln in range(layer_num):
        layer_wb = model.layers[ln].get_weights()
        weights.append(layer_wb[0].T)
        bias.append(layer_wb[1])

    num_inputs = 50
    num_outputs = 2

    print(layer_num,num_inputs,num_outputs,max(num_inputs,num_outputs,max(list(map(len,weights)))),'',sep=',')
    print(str(num_inputs)+','+','.join(list(map(lambda x: str(len(x)),weights)))+',')
    print('0,'*layer_num) # 0 = dense
    
    for layer in range(layer_num):
        for i in range(weights[layer].shape[0]):
            for j in range(weights[layer].shape[1]):
                print(str(weights[layer][i,j])+',', end = "")
            print("")
        for i in range(bias[layer].shape[0]):
            print(bias[layer][i], end="")
            print(",")
    
