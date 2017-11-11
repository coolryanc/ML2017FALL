import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, BatchNormalization, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam, RMSprop
from keras.models import load_model
from keras.callbacks import EarlyStopping
import utils
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix

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
        tmp /= 255
        train_data[index] = tmp.reshape(WIDTH, HEIGHT, 1)
    train_label = np_utils.to_categorical(train_label, num_classes=7)
    print()
    return train_data, train_label

if __name__ == '__main__':
    # train_data, train_label = readTrainingData()
    # np.save('./data/train_data.npy', train_data)
    # np.save('./data/train_label.npy', train_label)
    print("Read training data and label ... ")
    train_data = np.load('./data/train_data.npy')
    train_label = np.load('./data/train_label.npy')

    train_data_test = train_data[int(len(train_data)*9/10):]
    train_label_test = train_label[int(len(train_label)*9/10):]

    train_data = train_data[:int(len(train_data)*9/10)]
    train_label = train_label[:int(len(train_label)*9/10)]

    # OUTPUT_SIZE = train_label.shape[2] # 48phone_char
    ITERATION = 30
    BATCH_SIZE = 100

    model = Sequential()
    model.add(Flatten(input_shape=(train_data.shape[1], train_data.shape[2], train_data.shape[3])))

    model.add(Dense(1440, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(768, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(7))
    model.add(Activation('softmax'))

    optimizer = RMSprop()
    model.compile(loss='categorical_crossentropy',
                  optimizer= optimizer,
                  metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='dnn_model.png')
    # # early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    train_history = model.fit(train_data, train_label, epochs=ITERATION, batch_size=BATCH_SIZE, verbose=1, validation_data=(train_data_test, train_label_test))
    #
    score = model.evaluate(train_data_test, train_label_test)
    print("Loss: {}".format(score[0]))
    print("Accuracy: {}".format(score[1]))
    #
    y_pred = model.predict_classes(train_data_test)
    class_names = ['Angry', 'Digust', 'Fear', 'Happy', 'Sad', 'Suprise', 'Neutral']
    cnf_matrix = confusion_matrix(np.argmax(train_label_test,axis=1), y_pred)
    utils.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

    model.save('./model/model_dnn.h5')
    utils.show_train_history(train_history, 'acc', 'val_acc', 'acc.png')
    utils.show_train_history(train_history, 'loss', 'val_loss', 'loss.png')
    # del model
