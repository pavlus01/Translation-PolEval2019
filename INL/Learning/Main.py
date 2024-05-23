import warnings

import gensim
from keras.src.preprocessing import sequence
from keras.src.utils import np_utils

warnings.filterwarnings("ignore")
import re
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
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers
import matplotlib.pyplot as plt
plt.get_backend()
#optional if you want to generate statistical graphs of the DMT
#import matplotlib.pyplot as plt
#from keras.utils import plot_model
#import pydot

from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from keras_self_attention import SeqSelfAttention
from gensim.models import KeyedVectors

LEN_RU = 512


def main():
    model = Sequential()

    model_w2v = KeyedVectors.load_word2vec_format("../W2V_Models/kgr_mwe/v300/skip_gram_v300m8.w2v.txt", binary=False)
    # model_w2v = Word2Vec(common_texts, vector_size=100, window=5, min_count=1, workers=4)
    word_vectors = model_w2v.vectors
    vocab_size = len(word_vectors)
    embedding_dim = model_w2v.vector_size
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for i in range(vocab_size):
        embedding_matrix[i] = word_vectors[i]

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

    dfPl = read_file('../Data/MultiCCAligned.pl-ru.pl', 'PL')
    dfRu = read_file('../Data/MultiCCAligned.pl-ru.ru', 'RU')

    X_train, X_test, y_train, y_test = train_test_split(dfPl, dfRu)

    tokenizer = tokenization(X_train, vocab_size)

    sequences_trainX = tokenizer.texts_to_sequences(X_train)
    sequences_trainY = tokenizer.texts_to_sequences(X_train)
    sequences_testX = tokenizer.texts_to_sequences(X_test)
    sequences_testY = tokenizer.texts_to_sequences(X_test)

    X_train = sequence.pad_sequences(sequences_trainX, maxlen=LEN_RU)
    Y_train = sequence.pad_sequences(sequences_trainY, maxlen=LEN_RU)
    X_test = sequence.pad_sequences(sequences_testX, maxlen=LEN_RU)
    Y_test = sequence.pad_sequences(sequences_testY, maxlen=LEN_RU)

    epochs = 5
    batch_size = 7

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    rnn = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, shuffle=True)

    score = model.evaluate(X_test, Y_test)
    print("Test Loss: %.2f%%" % (score[0] * 100))
    print("Test Accuracy: %.2f%%" % (score[1] * 100))

    Plots(rnn, epochs)


def read_file(path, column_name):
    with open(path, 'r', encoding='utf-8') as file:
        lines = []
        for line in file:
            lines.append(line.strip())

    return pd.DataFrame(lines, columns=[column_name])


def train_test_split(X, y, test_size=0.2, random_state=None):
    assert len(X) == len(y), "X and y must have the same number of samples"

    # Ustawienie random seed dla powtarzalności wyników
    if random_state is not None:
        np.random.seed(random_state)

    # Liczba próbek
    num_samples = len(X)

    # Oblicz liczba próbek w zbiorze testowym
    num_test_samples = int(num_samples * test_size)

    # Wymieszanie indeksów
    indices = np.random.permutation(num_samples)

    # Podział na indeksy treningowe i testowe
    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]

    # Tworzenie zbiorów treningowych i testowych
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_test, y_train, y_test


def tokenization(lines, vocab_size):
    #print(lines)
    tokenizer = Tokenizer(num_words=vocab_size)

    tokenizer.fit_on_texts(lines)
    return tokenizer


def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq

def Plots(rnn, nb_epoch):
    plt.figure(0)
    print(rnn.history)
    plt.plot(rnn.history['accuracy'], 'r')
    plt.xticks(np.arange(0, nb_epoch + 1, nb_epoch / 5))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy LSTM l=10, epochs=20")  # for max length = 10 and 20 epochs
    plt.legend(['train', 'validation'])

    plt.figure(1)
    plt.plot(rnn.history['loss'], 'r')
    plt.xticks(np.arange(0, nb_epoch + 1, nb_epoch / 5))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Training vs Validation Loss LSTM l=10, epochs=20")  # for max length = 10 and 20 epochs
    plt.legend(['train', 'validation'])
    plt.show()
    plt.show(block=True)


# def learining():
#     temp = []
#     for j in range(len(i)):
#         t = get_word(i[j], ru_tokenizer)
#         if j > 0:
#             if (t == get_word(i[j - 1], ru_tokenizer)) or (t == None):
#                 temp.append('')
#             else:
#                 temp.append(t)
#         else:
#             if (t == None):
#                 temp.append('')
#             else:
#                 temp.append(t)
#     return ' '.join(temp)


if __name__ == "__main__":
    main()
