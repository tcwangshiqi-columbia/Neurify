import sys
import os
# quick fix to specify the GPUs to use, comment out if there is no one using 100% of them
os.environ["CUDA_VISIBLE_DEVICES"]="0,2"

import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Embedding, Dense, Dropout, Conv1D, Conv2D, Flatten, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, BatchNormalization  #create model
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Need weight path.')
        sys.exit(0)

    model = load_model(sys.argv[1])

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
    
