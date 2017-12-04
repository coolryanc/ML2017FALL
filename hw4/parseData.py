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
    # s = s.lower().strip()
    s = re.sub(r"([.!?])", r"", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def gen_sequence(label_x):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(label_x)
    word_num = len(tokenizer.word_index) + 1

    label_sequence = tokenizer.texts_to_sequences(label_x)
    # test_sequence = tokenizer.texts_to_sequences(test_data)

    label_sequence = pad_sequences(label_sequence)
    sequence_length = label_sequence.shape[1]
    # test_sequence = pad_sequences(test_sequence, maxlen=sequence_length)

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

# def readTrainingData(dataPath):
#     training_label = []
#     training_data = []
#     training_data_words = []
#     word_dic = {}
#     # Max length of sentence: 36
#     # Average length of sentence: 13.01299
#     word_index = 0
#     # SENTENCE_L = 16
#     with open(dataPath+'training_label.txt','r') as readFile:
#         lines = readFile.read().splitlines()
#         for index, line in enumerate(lines):
#             print('\rParse training data, line {}'.format(index+1), end='', flush=True)
#             line = line.split(' +++$+++ ')
#             words = normalizeString(line[1]).split(' ')
#             for w_i, w in enumerate(words):
#                 words[w_i] = stem(w)
#                 if words[w_i] not in word_dic:
#                     word_dic[words[w_i]] = word_index
#                     word_index += 1
#             training_label.append(line[0])
#             training_data_words.append(words)
#         training_label = np.array(training_label)
#         print('')
#         max_s = 0
#         count = 0
#         for index, sentence in enumerate(training_data_words):
#             print('\rConvert training data to embedding, line {}'.format(index+1), end='', flush=True)
#             tmp = []
#             if len(sentence) > max_s:
#                 max_s = len(sentence)
#             count += len(sentence)
#             for words in sentence:
#                 tmp.append(word_dic[words])
#             training_data.append(np.array(tmp))
#         training_data = np.array(training_data)
#         print("Max length of sentences: {}".format(max_s))
#         print("Average length: {}".format(count/len(training_data)))
#         return training_data, training_label
#
# def readTestingData(dataPath):
#     word_dic = {}
#     word_index = 0
#     with open(dataPath+'training_label.txt','r') as readFile:
#         lines = readFile.read().splitlines()
#         for index, line in enumerate(lines):
#             print('\rBuilt dic, line {}'.format(index+1), end='', flush=True)
#             line = line.split(' +++$+++ ')
#             words = normalizeString(line[1]).split(' ')
#             for w_i, w in enumerate(words):
#                 words[w_i] = stem(w)
#                 if words[w_i] not in word_dic:
#                     word_dic[words[w_i]] = word_index
#                     word_index += 1
#     print('')
#     test_data = []
#     with open(dataPath+'testing_data.txt','r') as readFile:
#         lines = readFile.read().splitlines()
#         lines = lines[1:] # ignore id, text
#         for index, line in enumerate(lines):
#             print('\rParse testing data, line {}'.format(index+1), end='', flush=True)
#             start = line.find(',')
#             words = normalizeString(line[start+1:]).split(' ')
#             tmp = []
#             for w in words:
#                 if w in word_dic:
#                     tmp.append(word_dic[w])
#             test_data.append(np.array(tmp))
#     test_data = np.array(test_data)
#     print('')
#     print(test_data.shape)
#     return test_data
