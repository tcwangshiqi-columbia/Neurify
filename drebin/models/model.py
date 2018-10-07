
from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.layers import Dense, Input, Activation
from keras.models import Model
from keras.optimizers import RMSprop
import numpy as np
from configs import bcolors

'''
in_placeholder = Input(shape=(545334,))
x = Dense(50, activation='relu')(in_placeholder)
x = Dense(50, activation='relu')(x)
x = Dense(2, name='before_softmax')(x)
x = Activation('softmax', name='predictions')(x)
model = Model(in_placeholder, x)

model.load_weights('./Model2.h5')
print(bcolors.OKBLUE + 'FC model loaded' + bcolors.ENDC)
'''
'''
in_placeholder = Input(shape=(545334,))
x = Dense(200, activation='relu')(in_placeholder)
x = Dense(200, activation='relu')(x)
x = Dense(2, name='before_softmax')(x)
x = Activation('softmax', name='predictions')(x)
model = Model(in_placeholder, x)
'''
in_placeholder = Input(shape=(545334,))
x = Dense(200, activation='relu')(in_placeholder)
x = Dense(10, activation='relu')(x)
x = Dense(2, name='before_softmax')(x)
x = Activation('softmax', name='predictions')(x)
model = Model(in_placeholder, x)

model.load_weights('./Model3.h5')
print(bcolors.OKBLUE + 'FC model loaded' + bcolors.ENDC)


weights = [model.get_weights()[0].T, model.get_weights()[2].T, model.get_weights()[4].T]
bias = [model.get_weights()[1], model.get_weights()[3], model.get_weights()[5]]
layer_Num = 3;

print ("3,545334,2,545334,")
print ("545334,200,200,2,")
for layer in range(layer_Num):
    for i in range(weights[layer].shape[0]):
        for j in range(weights[layer].shape[1]):
            print(str(weights[layer][i,j])+',', end = "")
        print("")
    for i in range(bias[layer].shape[0]):
        print(bias[layer][i], end="")
        print(",")