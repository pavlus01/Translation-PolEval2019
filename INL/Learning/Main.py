import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
from keras_self_attention import SeqSelfAttention
from keras.utils import Sequence

LEN_RU = 128  # Zmniejszenie długości sekwencji
LEN_PL = 128  # Zmniejszenie długości sekwencji
BATCH_SIZE = 2  # Zmniejszenie batch size
VOCAB_SIZE = 20000  # Ograniczenie rozmiaru słownika


def main():
    model = Sequential()

    model_w2v = KeyedVectors.load_word2vec_format("../W2V_Models/kgr_mwe/v100/skip_gram_v100m8.w2v.txt", binary=False)
    word_vectors = model_w2v.vectors
    embedding_dim = model_w2v.vector_size
    embedding_matrix = np.zeros((VOCAB_SIZE, embedding_dim))

    for i in range(min(VOCAB_SIZE, len(word_vectors))):
        embedding_matrix[i] = word_vectors[i]

    embedding_layer = Embedding(input_dim=VOCAB_SIZE,
                                output_dim=embedding_dim,
                                weights=[embedding_matrix],
                                trainable=False)

    model.add(embedding_layer)
    model.add(LSTM(128))  # Zmniejszenie liczby jednostek LSTM
    model.add(RepeatVector(LEN_RU))

    model.add(SeqSelfAttention(attention_activation='sigmoid'))

    model.add(LSTM(128, return_sequences=True))  # Zmniejszenie liczby jednostek LSTM
    model.add(Dense(VOCAB_SIZE, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    model.summary()

    dfPl = read_file('../Data/MultiCCAligned.pl-ru.pl', 'PL')
    dfRu = read_file('../Data/MultiCCAligned.pl-ru.ru', 'RU')

    X_train, X_test, y_train, y_test = train_test_split(dfPl, dfRu)

    tokenizer_pl = tokenization(X_train['PL'], VOCAB_SIZE)
    tokenizer_ru = tokenization(y_train['RU'], VOCAB_SIZE)

    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    train_generator = DataGenerator(X_train['PL'], y_train['RU'], BATCH_SIZE, tokenizer_pl, LEN_PL)
    validation_generator = DataGenerator(X_test['PL'], y_test['RU'], BATCH_SIZE, tokenizer_pl, LEN_PL)

    epochs = 5

    rnn = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    score = model.evaluate(validation_generator)
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


class DataGenerator(Sequence):
    def __init__(self, texts, labels, batch_size, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.indices = np.arange(len(texts))

    def __len__(self):
        return int(np.ceil(len(self.texts) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        print(f"Batch indices: {batch_indices}")  # Dodane logowanie
        batch_texts = self.texts.iloc[batch_indices].tolist()
        batch_labels = self.labels.iloc[batch_indices].tolist()

        X = pad_sequences(self.tokenizer.texts_to_sequences(batch_texts), maxlen=self.max_len)
        Y = pad_sequences(self.tokenizer.texts_to_sequences(batch_labels), maxlen=self.max_len)
        Y = np.expand_dims(Y, -1)

        return X, Y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


if __name__ == "__main__":
    main()
