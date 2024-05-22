import warnings
warnings.filterwarnings("ignore")
import numpy as np
import string
from numpy import array, argmax, random, take
#for processing imported data
import tensorflow as tf
import pandas as pd
#the RNN routines
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers
#optional if you want to generate statistical graphs of the DMT
#import matplotlib.pyplot as plt
#from keras.utils import plot_model
#import pydot

from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from keras_self_attention import SeqSelfAttention

def main():
    LEN_RU = 512

    model = Sequential()

    model_w2v = Word2Vec(common_texts, vector_size=100, window=5, min_count=1, workers=4)
    word_vectors = model_w2v.wv
    vocab_size = len(word_vectors)
    embedding_dim = word_vectors.vector_size
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for i in range(vocab_size):
        embedding_matrix[i] = word_vectors[word_vectors.index_to_key[i]]

    embedding_layer = Embedding(input_dim=vocab_size,
                                output_dim=embedding_dim,
                                weights=[embedding_matrix],
                                trainable=False)

    model.add(embedding_layer)
    model.add(LSTM(512))
    model.add(RepeatVector(8))

    model.add(SeqSelfAttention(attention_activation='sigmoid'))

    model.add(LSTM(512))
    model.add(Dense(LEN_RU, activation='softmax'))
    rms = optimizers.RMSprop(lr=0.001)
    model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')

    #plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)

    model.summary()

def tokenization(lines):
        #print(lines)
        tokenizer = Tokenizer()

        tokenizer.fit_on_texts(lines)
        return tokenizer

def encode_sequences(tokenizer, length, lines):
         # integer encode sequences
         seq = tokenizer.texts_to_sequences(lines)
         # pad sequences with 0 values
         seq = pad_sequences(seq, maxlen=length, padding='post')
         return seq

def learining():
    temp = []
    for j in range(len(i)):
        t = get_word(i[j], ru_tokenizer)
        if j > 0:
            if (t == get_word(i[j - 1], ru_tokenizer)) or (t == None):
                temp.append('')
            else:
                temp.append(t)
        else:
            if (t == None):
                temp.append('')
            else:
                temp.append(t)
    return ' '.join(temp)


if __name__ == "__main__":
    main()