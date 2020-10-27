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

import re

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
# normalise
x = (x+1)/2
x = x.flatten()

out_string = ','.join([str(i) for i in x])
print(out_string+',')
