# if __name__ == '__main__':
#     print('Hello World')

import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
import string
from numpy import array, argmax, random, take
#for processing imported data
import pandas as pd
#the RNN routines
from keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, RepeatVector
#we will need the tokenizer for BERT
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers


def main():
    print("hey there")