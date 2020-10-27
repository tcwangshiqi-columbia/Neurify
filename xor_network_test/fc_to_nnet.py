import tensorflow as tf
from tensorflow import keras
from maraboupy import Marabou
from maraboupy import MarabouUtils, MarabouCore
from tensorflow.keras.utils import to_categorical  # one-hot encode target column
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, Dense, Dropout, Conv1D, Conv2D, Flatten, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, BatchNormalization  #create model
import numpy as np
import sys

model = Sequential()
model.add(Dense(3, input_shape=(2,), activation='relu'))
model.add(Dense(2,name='before_softmax'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

model.load_weights('xor.h5')

layer_num = len(model.layers)-1 # before softmax layer
weights = []
bias = []
for ln in range(layer_num):
    layer_wb = model.layers[ln].get_weights()
    weights.append(layer_wb[0].T)
    bias.append(layer_wb[1])

num_inputs = 2
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
