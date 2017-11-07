import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense, LSTM, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.models import load_model
from collections import OrderedDict
import itertools
import sys, os

def readTrainingData():
    WIDTH = HEIGHT = 48
    print('Parsing Training Data ...')
    raw_data = pd.read_csv('./data/test.csv').as_matrix() #shape: (28709, 2)
    testing_data = np.zeros((len(raw_data[:, 1]), WIDTH, HEIGHT, 1)) # init
    for index, i in enumerate(raw_data[:, 1]):
        print('\rIteration: {}, Split testing data to 48x48x1'.format(index), end='', flush=True)
        tmp = np.array(i.split(' '))
        tmp = tmp.astype(np.float)
        tmp /= 255
        testing_data[index] = tmp.reshape(WIDTH, HEIGHT, 1)
    print()
    return testing_data

def getResultLabel(result):
    concert_result = []
    for i in result:
        origin_label = np.where(max(i)==i)[0][0]
        concert_result.append(origin_label)
    return concert_result

def main(argv):
    testing_data = readTrainingData()

    model = load_model('./model/model_cnn.h5')
    # print("Start reading model ...")
    # if argv[1] == "rnn" or argv[1] == "best":
    #     model = load_model('./model/model_rnn.h5')
    # elif argv[1] == "cnn":
    #     model = load_model('./model/model_cnn.h5')
    print("Predict ...")

    result = model.predict(testing_data, batch_size=100, verbose=0)
    writeText = "id,label\n"
    result = getResultLabel(result[:,:])
    for index, r in enumerate(result):
        writeText += str(index) + "," + str(r) + '\n'
    filename = './result.csv'
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(writeText)

if __name__ == '__main__':
    main(sys.argv)
