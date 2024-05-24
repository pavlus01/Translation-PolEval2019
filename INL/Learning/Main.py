import warnings
import gensim
from keras.src.preprocessing import sequence
from keras.src.utils import np_utils
warnings.filterwarnings("ignore")
import re
import numpy as np
import string
from numpy import array, argmax, random, take
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from keras_self_attention import SeqSelfAttention

LEN_RU = 512
LEN_PL = 512

def main():
    model = Sequential()

    model_w2v = KeyedVectors.load_word2vec_format("../W2V_Models/kgr_mwe/v100/skip_gram_v100m8.w2v.txt", binary=False)
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
    model.add(RepeatVector(LEN_RU))

    model.add(SeqSelfAttention(attention_activation='sigmoid'))

    model.add(LSTM(512, return_sequences=True))
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    model.summary()

    dfPl = read_file('../Data/MultiCCAligned.pl-ru.pl', 'PL')
    dfRu = read_file('../Data/MultiCCAligned.pl-ru.ru', 'RU')

    X_train, X_test, y_train, y_test = train_test_split(dfPl, dfRu)

    tokenizer_pl = tokenization(X_train['PL'], vocab_size)
    tokenizer_ru = tokenization(y_train['RU'], vocab_size)

    sequences_trainX = tokenizer_pl.texts_to_sequences(X_train['PL'])
    sequences_trainY = tokenizer_ru.texts_to_sequences(y_train['RU'])
    sequences_testX = tokenizer_pl.texts_to_sequences(X_test['PL'])
    sequences_testY = tokenizer_ru.texts_to_sequences(y_test['RU'])

    X_train = pad_sequences(sequences_trainX, maxlen=LEN_PL)
    Y_train = pad_sequences(sequences_trainY, maxlen=LEN_RU)
    X_test = pad_sequences(sequences_testX, maxlen=LEN_PL)
    Y_test = pad_sequences(sequences_testY, maxlen=LEN_RU)

    Y_train = np.expand_dims(Y_train, -1)
    Y_test = np.expand_dims(Y_test, -1)

    epochs = 5
    batch_size = 7

    rnn = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, shuffle=True)

    score = model.evaluate(X_test, Y_test)
    print("Test Loss: %.2f%%" % (score[0] * 100))
    print("Test Accuracy: %.2f%%" % (score[1] * 100))

    df = pd.DataFrame({'PL': ["Kocham moją mamę"]})
    single_value = pad_sequences(tokenizer_pl.texts_to_sequences(df['PL']), maxlen=LEN_PL)

    prediction = model.predict(single_value)
    prediction = np.argmax(prediction, axis=-1)

    reverse_word_map_ru = dict(map(reversed, tokenizer_ru.word_index.items()))
    decoded_sentence = ' '.join([reverse_word_map_ru.get(i, '?') for i in prediction[0]])

    print(f'Predykcja dla podanej wartości: {decoded_sentence}')


def read_file(path, column_name):
    with open(path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file]
    return pd.DataFrame(lines, columns=[column_name])


def train_test_split(X, y, test_size=0.2, random_state=None):
    assert len(X) == len(y), "X and y must have the same number of samples"
    if random_state is not None:
        np.random.seed(random_state)
    num_samples = len(X)
    num_test_samples = int(num_samples * test_size)
    indices = np.random.permutation(num_samples)
    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    return X_train, X_test, y_train, y_test


def tokenization(lines, vocab_size):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(lines)
    return tokenizer

def encode_sequences(tokenizer, length, lines):
    seq = tokenizer.texts_to_sequences(lines)
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq

def Plots(rnn, nb_epoch):
    plt.figure(0)
    plt.plot(rnn.history['accuracy'], 'r')
    plt.xticks(np.arange(0, nb_epoch + 1, nb_epoch / 5))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy LSTM l=10, epochs=20")
    plt.legend(['train', 'validation'])
    plt.figure(1)
    plt.plot(rnn.history['loss'], 'r')
    plt.xticks(np.arange(0, nb_epoch + 1, nb_epoch / 5))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Training vs Validation Loss LSTM l=10, epochs=20")
    plt.legend(['train', 'validation'])
    plt.show()
    plt.show(block=True)

if __name__ == "__main__":
    main()
