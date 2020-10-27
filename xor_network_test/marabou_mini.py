import tensorflow as tf
from tensorflow import keras
from maraboupy import Marabou
from maraboupy import MarabouUtils, MarabouCore
from tensorflow.keras.utils import to_categorical  # one-hot encode target column
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, Dense, Dropout, Conv1D, Conv2D, Flatten, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, BatchNormalization  #create model
import numpy as np

'''
2 input, 3 hidden, 2 output
h0_w = [0.1,-0.2]
h1_w = [0.5,0.3]
h2_w = [-0.6,-0.1]

o0_w = [0.3,-0.2,0.6]
o1_w = [-0.2,-0.1,0.5]
'''
model = Sequential()
model.add(Dense(3, input_shape=(2,), activation='relu'))
model.add(Dense(2,name='before_softmax'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

model.load_weights('xor.h5')

model_before_softmax = tf.keras.models.Model(inputs=model.inputs,outputs=model.layers[-2].output)
tf.saved_model.save(model_before_softmax,'test_model/')
filename = 'test_model/'

network = Marabou.read_tf(filename,modelType='savedModel_v2',savedModelTags=['serving_default'])

inputVars = network.inputVars[0][0].flatten()
outputVars = network.outputVars[0].flatten()

with open('input_target_1','r') as f:
    file_inputs = f.read().split(',')[:-1]
    file_inputs = list(map(float,file_inputs))
    target = 1
    opp = 0

eps = 0.18
for idx,i in enumerate(inputVars):
    lower = max(-1.0,file_inputs[idx]-eps)
    upper = min(1.0,file_inputs[idx]+eps)
    network.setLowerBound(i,lower)
    network.setUpperBound(i,upper)

network.addInequality([outputVars[target],outputVars[opp]],[1,-1],-1e-3)
opts = Marabou.createOptions(verbosity=2)
vals, _ = network.solve(options=opts,verbose=True)
print(vals)
