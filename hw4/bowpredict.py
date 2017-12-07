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
    X_train, y_train, testing_data, word_num, sequence_length  = parseData.bowSequence('./data/')

    print("Start reading model ...")
    model = load_model('./model/bowbest.h5')
    print("Predict ...")
    print(testing_data.shape)
    tmpResult = model.predict(testing_data, verbose=0)

    model1 = load_model('./model/bowbest1.h5')
    print("Predict ...")
    tmpResult1 = model1.predict(testing_data, verbose=0)

    tmpResult = np.array(tmpResult)
    tmpResult1 = np.array(tmpResult1)

    result = tmpResult + tmpResult1
    result /= 2

    writeText = "id,label\n"
    for i, ans in enumerate(result):
        if float(ans) > 0.5:
            ans = 1
        else:
            ans = 0
        writeText += str(i) + ',' + str(ans) + '\n'
    filename = './t.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(writeText)

if __name__ == '__main__':
    main(sys.argv)
