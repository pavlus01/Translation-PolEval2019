import os
import sys

import keras.saving
from keras.utils import plot_model, to_categorical, Sequence
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray, zeros

BATCH_SIZE = 35
EPOCHS = 20
LSTM_NODES = 256
NUM_SENTENCES = 10000
MAX_SENTENCE_LENGTH = 50
MAX_NUM_WORDS = 42000
EMBEDDING_SIZE = 100


class DataGenerator(Sequence):
    def __init__(self, input_sequences, output_sequences, batch_size, num_words_output, max_out_len):
        self.input_sequences = input_sequences
        self.output_sequences = output_sequences
        self.batch_size = batch_size
        self.num_words_output = num_words_output
        self.max_out_len = max_out_len

    def __len__(self):
        return int(np.ceil(len(self.input_sequences) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.input_sequences[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.output_sequences[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_decoder_targets = np.zeros((len(batch_x), self.max_out_len, self.num_words_output), dtype='float32')

        for i, d in enumerate(batch_y):
            for t, word in enumerate(d):
                if word < self.num_words_output:
                    batch_decoder_targets[i, t, word] = 1

        return [batch_x, pad_sequences(batch_y, maxlen=self.max_out_len, padding='post')], batch_decoder_targets


def main():
    input_sentences, output_sentences, output_sentences_inputs = data_read_preparation()

    input_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    input_tokenizer.fit_on_texts(input_sentences)
    input_integer_seq = input_tokenizer.texts_to_sequences(input_sentences)

    word2idx_inputs = input_tokenizer.word_index
    print('Total unique words in the input:', len(word2idx_inputs))

    max_input_len = max(len(sen) for sen in input_integer_seq)
    print("Length of longest sentence in input:", max_input_len)

    output_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
    output_tokenizer.fit_on_texts(output_sentences + output_sentences_inputs)
    output_integer_seq = output_tokenizer.texts_to_sequences(output_sentences)
    output_input_integer_seq = output_tokenizer.texts_to_sequences(output_sentences_inputs)

    word2idx_outputs = output_tokenizer.word_index
    print('Total unique words in the output:', len(word2idx_outputs))

    num_words_output = len(word2idx_outputs) + 1
    max_out_len = max(len(sen) for sen in output_integer_seq)
    print("Length of longest sentence in the output:", max_out_len)

    encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=max_input_len)

    decoder_input_sequences = pad_sequences(output_input_integer_seq, maxlen=max_out_len, padding='post')

    embeddings_dictionary = dict()

    glove_file = open(r'../W2V_Models/kgr_mwe/v100/cbow_v100m8_hs.w2v.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()

    num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
    embedding_matrix = zeros((num_words, EMBEDDING_SIZE))
    for word, index in word2idx_inputs.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    embedding_layer = Embedding(num_words, EMBEDDING_SIZE, weights=[embedding_matrix], input_length=max_input_len)

    encoder_inputs_placeholder = Input(shape=(max_input_len,))
    x = embedding_layer(encoder_inputs_placeholder)
    encoder = LSTM(LSTM_NODES, return_state=True)

    encoder_outputs, h, c = encoder(x)
    encoder_states = [h, c]

    decoder_inputs_placeholder = Input(shape=(max_out_len,))
    decoder_embedding = Embedding(num_words_output, LSTM_NODES)
    decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

    decoder_lstm = LSTM(LSTM_NODES, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)

    decoder_dense = Dense(num_words_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs)
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # model = keras.saving.load_model("../Models/model.keras")

    encoder_model = keras.saving.load_model("../Models/encoder.keras")
    decoder_model = keras.saving.load_model("../Models/decoder.keras")

    data_generator = DataGenerator(encoder_input_sequences, output_integer_seq, BATCH_SIZE, num_words_output,
                                   max_out_len)
    validation_data_generator = DataGenerator(encoder_input_sequences[:int(NUM_SENTENCES * 0.1)],
                                              output_integer_seq[:int(NUM_SENTENCES * 0.1)], BATCH_SIZE,
                                              num_words_output, max_out_len)

    r = model.fit(
        data_generator,
        steps_per_epoch=NUM_SENTENCES/BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_data_generator,
        validation_steps=len(validation_data_generator),
    )

    encoder_model = Model(encoder_inputs_placeholder, encoder_states)

    decoder_state_input_h = Input(shape=(LSTM_NODES,))
    decoder_state_input_c = Input(shape=(LSTM_NODES,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_inputs_single = Input(shape=(1,))
    decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

    decoder_outputs, h, c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)
    decoder_states = [h, c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs_single] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    idx2word_input = {v: k for k, v in word2idx_inputs.items()}
    idx2word_target = {v: k for k, v in word2idx_outputs.items()}

    i = np.random.choice(10)
    input_seq = encoder_input_sequences[i:i + 1]
    translation = translate_sentence(input_seq, encoder_model, word2idx_outputs, max_out_len, decoder_model,
                                     idx2word_target)
    print('-')
    print('Input:', input_sentences[i])
    print('Response:', translation)

    model.save("../Models/model.keras")
    encoder_model.save("../Models/encoder.keras")
    decoder_model.save("../Models/decoder.keras")


def translate_sentence(input_seq, encoder_model, word2idx_outputs, max_out_len, decoder_model, idx2word_target):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<eos>']
    output_sentence = []

    for _ in range(max_out_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''

        if idx > 0:
            word = idx2word_target[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(output_sentence)


def data_read_preparation():
    input_sentences = []
    output_sentences = []
    output_sentences_inputs = []

    count = 0
    for line in open(r'../Data/TED2020.es-pl.txt', encoding="utf-8"):
        count += 1

        if count > NUM_SENTENCES:
            break

        if '##' not in line:
            continue

        input_sentence, output = line.rstrip().split('##')

        output_sentence = output + ' <eos>'
        output_sentence_input = '<sos> ' + output

        input_sentences.append(input_sentence)
        output_sentences.append(output_sentence)
        output_sentences_inputs.append(output_sentence_input)

    print("num samples input:", len(input_sentences))
    print("num samples output:", len(output_sentences))
    print("num samples output input:", len(output_sentences_inputs))

    print(input_sentences[172])
    print(output_sentences[172])
    print(output_sentences_inputs[172])

    return input_sentences, output_sentences, output_sentences_inputs


if __name__ == "__main__":
    main()
