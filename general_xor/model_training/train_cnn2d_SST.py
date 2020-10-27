"""
Simple yet useful blob to train all the models from IMDB, SST and AG datasets
- TODO: make all these functions dynamic and merge shared code
"""

import numpy as np
import string
import tensorflow as tf
from pandas import read_csv
from tensorflow.keras.datasets import imdb  # use the same words index of the IMDB dataset
from tensorflow.keras.utils import to_categorical  # one-hot encode target column
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Embedding, Dense, Dropout, Conv1D, Conv2D, Flatten, GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization  #create model

from glove_utils import load_embedding, pad_sequences
from text_utils import clean_text, stem_lem

# Global model variables
maxlen = 9
ksize = int(maxlen**0.5)
emb_dims = 5
kernel_dim = 2  # 'window' size
padding = "valid"  # "valid", "causal" or "same"
stride = 1
INDEX_FROM = 3   # word index offset
epochs = 5
round_ = None  # number of digits to round input (i.e., embedding), no round if None

# Load STT dataset (eliminate punctuation, add padding etc.)
print("[logger]: building STT custom model with tf version {}".format(tf.__version__))
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
model.add(Conv2D(32, (3,3), padding="same",
                 input_shape=(ksize, ksize, emb_dims), activation='relu'))
model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

# Prepare test set in advance
X_test = [[index2embedding[word2index[x]] for x in xx] for xx in X_test]
X_test = np.asarray(pad_sequences(X_test, maxlen=maxlen, emb_size=emb_dims))
X_test = X_test.reshape(len(X_test), ksize, ksize, emb_dims)
y_test = to_categorical(y_test, num_classes=2)

X_train_inp = [[index2embedding[word2index[x]] for x in xx] for xx in X_train]
X_train_inp = np.asarray(pad_sequences(X_train_inp, maxlen=maxlen, emb_size=emb_dims))
X_train_inp = X_train_inp.reshape(len(X_train_inp), ksize, ksize, emb_dims)
y_train_inp = to_categorical(y_train, num_classes=2)

model.fit(X_train_inp, y_train_inp, batch_size=512, epochs=100)
model.evaluate(X_test, y_test, batch_size=512)

model.save_weights('models/SST_cnn_5d_9inp_format_c_mp_2.h5')
