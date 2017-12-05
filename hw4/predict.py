import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense, LSTM, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.models import load_model
from collections import OrderedDict
from keras.preprocessing import sequence
import itertools
import sys, os
import parseData

def main(argv):
    testing_data = parseData.readTestingData('./data/')
    # max_review_length = 36
    # testing_data = sequence.pad_sequences(testing_data, maxlen=max_review_length)

    # print("Start reading model ...")
    model = load_model('./model/rnn_model_1.h5')
    # print("Predict ...")
    result = model.predict(testing_data, batch_size=64, verbose=0)
    writeText = "id,label\n"
    for i, ans in enumerate(result):
        if float(ans) > 0.5:
            ans = 1
        else:
            ans = 0
        writeText += str(i) + ',' + str(ans) + '\n'
    filename = './result1205.csv'
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(writeText)

if __name__ == '__main__':
    main(sys.argv)
