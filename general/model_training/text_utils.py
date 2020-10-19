import re
import string as string
from tensorflow.keras.datasets import imdb  # use the same words index of the IMDB dataset
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def clean_text(text, token='N'):
    text = text.lower()  # lower case words
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    return text

def stem_lem(words):
    stemmatizer, lemmatizer = PorterStemmer(), WordNetLemmatizer()
    words = [stemmatizer.stem(w) for w in words]  # stemming
    words = [lemmatizer.lemmatize(w) for w in words]  # lemmization
    return words

def pad_sequences(inputs, pad_token, maxlen):
    """
    Pad a list of texts (encoded as list of indices, one for each word) up to a maxlen parameter.
    """
    X = []
    for input_ in inputs:
        X.append([])
        if len(input_) > maxlen:
            X[-1] = input_[:maxlen]
        else:
            X[-1] = input_ + [pad_token for _ in range(maxlen-len(input_))]
    return X

def imdb2indices(inputs):
    """
    Turn a list of texts (encoded as list of words) into indices, according to the words
     that are present in the imdb dataset (as implemented by Keras).
    """
    X = []  # results
    word2index = imdb.get_word_index()
    word2index = {k:(v+3) for k,v in word2index.items()}
    word2index["<PAD>"], word2index["<START>"], word2index["<UNK>"], word2index["<UNUSED>"] = 0,1,2,3
    for input_ in inputs:
        X.append([])
        for word in input_:
            idx = word2index.get(word, word2index["<UNK>"])
            X[-1].append(idx)
    return X
