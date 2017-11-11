import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense, LSTM, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.models import load_model
import keras.backend as K
import itertools
import sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def readTrainingData():
    WIDTH = HEIGHT = 48
    print('Parsing Training Data ...')
    raw_data = pd.read_csv('./data/train.csv').as_matrix() #shape: (28709, 2)
    train_label = raw_data[:, 0]
    train_data = np.zeros((len(raw_data[:, 1]), WIDTH, HEIGHT, 1)) # init
    for index, i in enumerate(raw_data[:, 1]):
        print('\rIteration: {}, Split training data to 48x48x1'.format(index), end='', flush=True)
        tmp = np.array(i.split(' '))
        tmp = tmp.astype(np.float)
        train_data[index] = tmp.reshape(WIDTH, HEIGHT, 1)
    print()
    return train_data, train_label

def readTestingData(dataPath):
    WIDTH = HEIGHT = 48
    print('Parsing Testing Data ...')
    raw_data = pd.read_csv(dataPath).as_matrix() #shape: (28709, 2)
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

def plot_saliency_map():
    train_data, train_label = readTrainingData()

    X = train_data / 255
    _X  = train_data.astype('int')

    model = load_model('./model/model_cnn_v9.h5')
    input_img = model.input
    img_ids = [102,928,8377]

    for idx in img_ids:
        val_proba = model.predict(X[idx].reshape(-1, 48, 48, 1))
        pred = val_proba.argmax(axis=-1)
        target = K.mean(model.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])

        print("ID: {}, Label: {}, Predict: {}".format(idx, train_label[idx], pred))

        val_grads = fn([X[idx].reshape(-1, 48, 48, 1), 0])[0].reshape(48, 48, -1)

        val_grads *= -1
        val_grads = np.max(np.abs(val_grads), axis=-1, keepdims=True)

        # normalize
        val_grads = (val_grads - np.mean(val_grads)) / (np.std(val_grads) + 1e-30)
        val_grads *= 0.1

        # clip to [0, 1]
        val_grads += 0.5
        val_grads = np.clip(val_grads, 0, 1)

        # scale to [0, 1]
        val_grads /= np.max(val_grads)

        heatmap = val_grads.reshape(48, 48)

        # show original image
        plt.figure()
        plt.imshow(_X[idx].reshape(48, 48), cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join('./origin/', '{}.png'.format(idx)), dpi=100)

        thres = 0.5
        see = _X[idx].reshape(48, 48)
        see[np.where(heatmap <= thres)] = np.mean(see)

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join('./heatMap/', '{}.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow(see, cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join('./partial_see_dir/', '{}.png'.format(idx)), dpi=100)

def main(argv):
    testing_data = readTestingData(argv[1])

    model = load_model('./model/model_cnn_v9.h5')

    print("Predict ...")

    result = model.predict(testing_data, batch_size=100, verbose=0)
    writeText = "id,label\n"
    result = getResultLabel(result[:,:])
    for index, r in enumerate(result):
        writeText += str(index) + "," + str(r) + '\n'
    filename = argv[2]
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(writeText)

if __name__ == '__main__':
    main(sys.argv)
    # plot_saliency_map()
