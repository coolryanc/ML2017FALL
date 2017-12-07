from stemming.porter2 import stem
import re
import json
import unicodedata
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer

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

def bowSequence(dataPath):
    corpus = []
    training_label = []
    NUMSOFWORDS = 10000
    tokenizer = Tokenizer(num_words=NUMSOFWORDS, filters="")
    with open(dataPath+'training_label.txt','r') as f:
        for index, line in enumerate(f):
            if index % 1000 == 0:
                print('\rParse training data, line {}'.format(index), end='', flush=True)
            doc = []
            line = line.split(' +++$+++ ')
            words = normalizeString(line[1])
            training_label.append(line[0])
            corpus.append(words)
    tokenizer.fit_on_texts(corpus)
    training_bagOfDense = tokenizer.texts_to_matrix(corpus, mode='count')
    corpus = []
    with open(dataPath+'testing_data.txt','r') as readFile:
        lines = readFile.read().splitlines()
        lines = lines[1:] # ignore id, text
        for index, line in enumerate(lines):
            start = line.find(',')
            words = normalizeString(line[start+1:])
            corpus.append(words)
    test_bagOfDense = tokenizer.texts_to_matrix(corpus, mode='count')
    training_label = np.array(training_label)
    return training_bagOfDense, training_label, test_bagOfDense, NUMSOFWORDS, NUMSOFWORDS

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
    with open(dataPath,'r') as readFile:
        lines = readFile.read().splitlines()
        for index, line in enumerate(lines):
            if index % 1000 == 0:
                print('\rParse training data, line {}'.format(index), end='', flush=True)
            line = line.split(' +++$+++ ')
            words = normalizeString(line[1])
            training_label.append(line[0])
            training_data.append(words)
        label_sequence, word_num, sequence_length = gen_sequence(training_data)
        training_label = np.array(training_label)
        return label_sequence, word_num, sequence_length, training_label

def readTestingData(dataPath):
    test_data = []
    # training_data = []
    # with open('./data/training_label.txt','r') as readFile:
    #     lines = readFile.read().splitlines()
    #     for index, line in enumerate(lines):
    #         line = line.split(' +++$+++ ')
    #         words = normalizeString(line[1])
    #         training_data.append(words)
    # outputData = open("words.txt","w")
    # outputData.write(json.dumps(training_data))
    readfile = open("./words.txt",'r')
    training_data = json.loads(readfile.read())
    with open(dataPath,'r') as readFile:
        lines = readFile.read().splitlines()
        lines = lines[1:] # ignore id, text
        for index, line in enumerate(lines):
            start = line.find(',')
            words = normalizeString(line[start+1:])
            test_data.append(words)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(training_data)
        test_sequence = tokenizer.texts_to_sequences(test_data)
        test_sequence = pad_sequences(test_sequence, maxlen=36)
        return test_sequence

if __name__ == '__main__':
    # bowSequence('./data/')
    readTestingData('./data/testing_data.txt')
