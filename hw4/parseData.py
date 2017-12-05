from stemming.porter2 import stem
import re
import unicodedata
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r"", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def gen_sequence(label_x):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(label_x)
    word_num = len(tokenizer.word_index) + 1

    label_sequence = tokenizer.texts_to_sequences(label_x)
    label_sequence = pad_sequences(label_sequence)
    sequence_length = label_sequence.shape[1]

    return label_sequence, word_num, sequence_length

def readTrainingData(dataPath):
    training_label = []
    training_data = []
    with open(dataPath+'training_label.txt','r') as readFile:
        lines = readFile.read().splitlines()
        for index, line in enumerate(lines):
            print('\rParse training data, line {}'.format(index+1), end='', flush=True)
            line = line.split(' +++$+++ ')
            words = normalizeString(line[1])
            training_label.append(line[0])
            training_data.append(words)
        label_sequence, word_num, sequence_length = gen_sequence(training_data)
        training_label = np.array(training_label)
        return label_sequence, word_num, sequence_length, training_label

def readTestingData(dataPath):
    test_data = []
    training_data = []
    with open(dataPath+'training_label.txt','r') as readFile:
        lines = readFile.read().splitlines()
        for index, line in enumerate(lines):
            # print('\rParse training data, line {}'.format(index+1), end='', flush=True)
            line = line.split(' +++$+++ ')
            words = normalizeString(line[1])
            training_data.append(words)
    with open(dataPath+'testing_data.txt','r') as readFile:
        lines = readFile.read().splitlines()
        lines = lines[1:] # ignore id, text
        for index, line in enumerate(lines):
            # print('\rParse testing data, line {}'.format(index+1), end='', flush=True)
            start = line.find(',')
            words = normalizeString(line[start+1:])
            test_data.append(words)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(training_data)
        word_num = len(tokenizer.word_index) + 1
        test_sequence = tokenizer.texts_to_sequences(test_data)
        test_sequence = pad_sequences(test_sequence, maxlen=36)
        return test_sequence
