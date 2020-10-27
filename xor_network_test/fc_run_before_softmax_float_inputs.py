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
model = Sequential()
model.add(Dense(3, input_shape=(2,), activation='relu'))
model.add(Dense(2,name='before_softmax'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

model.load_weights('xor.h5')

with open('input_target_1','r') as f:
    text_in = f.read().split(',')[:-1]
text_in = list(map(float,text_in))
text_in = np.array([text_in])
# print(text_in)
# print(x)
x = text_in


extractor = tf.keras.models.Model(inputs=model.inputs,outputs=model.layers[-2].output)
features = extractor.predict(x)
print(features)
