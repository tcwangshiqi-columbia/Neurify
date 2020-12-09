"""
Simple yet useful blob to train all the models from IMDB, SST and AG datasets
- TODO: make all these functions dynamic and merge shared code
"""
import os
import sys

# quick fix to specify the GPUs to use, comment out if there is no one using 100% of them
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

import numpy as np
import string
import tensorflow as tf
from pandas import read_csv
from tensorflow.keras.datasets import imdb  # use the same words index of the IMDB dataset
from tensorflow.keras.utils import to_categorical  # one-hot encode target column
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, Dense, Dropout, Conv1D, Conv2D, Flatten, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, BatchNormalization  #create model

from glove_utils import load_embedding, pad_sequences
from text_utils import clean_text, stem_lem

# Global model variables
maxlen = 10
kernel_dim = 2
padding = "valid" # same = padding, valid = no padding
stride = 1
emb_dims = 5
square = int((maxlen*emb_dims)**0.5)
print('Input size:',square,'x',square)
in_shape = (square,square,1)
round_ = None  # number of digits to round input (i.e., embedding), no round if None
epochs = 50

# Load STT dataset (eliminate punctuation, add padding etc.)
X_train = read_csv('./data/SST_2__FULL.csv', sep=',',header=None).values
X_test = read_csv('./data/SST_2__TEST.csv', sep=',',header=None).values
y_train, y_test = [], []
for i in range(len(X_train)):
    r, s = X_train[i]  # review, score (comma separated in the original file)
    X_train[i][0] = [w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')]
    y_train.append((0 if s.strip()=='negative' else 1))
for i in range(len(X_test)):
    r, s = X_test[i]
    X_test[i][0] = [w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')]
    y_test.append((0 if s.strip()=='negative' else 1))
X_train, X_test = X_train[:,0], X_test[:,0]
n = -1  # you may want to take just some samples (-1 to take them all)
X_train = X_train[:n]
X_test = X_test[:n]
y_train = y_train[:n]
y_test = y_test[:n]

# Inputs as Numpy arrays
X_train = np.array([np.array(x) for x in X_train])
X_test = np.array([np.array(x) for x in X_test]) 

# Select the embedding
path_to_embeddings = './embeddings/'
EMBEDDING_FILENAME = path_to_embeddings+'custom-embedding-SST.{}d.txt'.format(emb_dims)
word2index, _, index2embedding = load_embedding(EMBEDDING_FILENAME)
if round_ is not None:
    print("[logger]: Quantized embedding to {} digits".format(round_))
    index2embedding = np.around(index2embedding, round_)

# Create the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=kernel_dim, strides=(stride,stride), padding=padding,
                 input_shape=in_shape, activation='relu'))
#model.add(MaxPooling2D())
'''
model.add(Conv2D(filters=300, kernel_size=kernel_dim, strides=stride, padding=padding,
                 activation='relu'))
model.add(MaxPooling2D())
'''
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(2,name='before_softmax'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

# Prepare test set in advance
X_test = [[index2embedding[word2index[x]] for x in xx] for xx in X_test]
X_test = np.asarray(pad_sequences(X_test, maxlen=maxlen, emb_size=emb_dims))
# normalise, Neurify requires >=0 inputs
# original range is [-1,1], turn it into [0,1]
X_test = (X_test+1)/2
X_test = X_test.reshape(len(X_test), *in_shape)

y_test = to_categorical(y_test, num_classes=2)

X_train_chunk = [[index2embedding[word2index[x]] for x in xx] for xx in X_train]        
X_train_chunk = np.asarray(pad_sequences(X_train_chunk, maxlen=maxlen, emb_size=emb_dims))
# normalise, Neurify requires >=0 inputs
# original range is [-1,1], turn it into [0,1]
X_train_chunk = (X_train_chunk+1)/2
X_train_chunk = X_train_chunk.reshape(len(X_train_chunk), *in_shape)

y_train_chunk = to_categorical(y_train, num_classes=2)
model.fit(X_train_chunk, y_train_chunk, batch_size=512, epochs=epochs)
acc = model.evaluate(X_test, y_test, batch_size=512)
acc = acc[1]

model_name = f'models/fullmodel_SST_conv_{emb_dims}d_{maxlen}inp_c32_f_d32_d16_d2_{acc:.4f}acc.h5'
print('Saved to:',model_name)
model.save(model_name)
