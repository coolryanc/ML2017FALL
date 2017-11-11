import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, BatchNormalization, Convolution2D, MaxPooling2D, Flatten, LeakyReLU
from keras.optimizers import Adam, RMSprop
from keras.models import load_model
from keras.callbacks import EarlyStopping
import utils
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import sys, os

def readTrainingData(dataPath):
    WIDTH = HEIGHT = 48
    print('Parsing Training Data ...')
    raw_data = pd.read_csv(dataPath).as_matrix() #shape: (28709, 2)
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
    train_data, train_label = readTrainingData(sys.argv[1])
    # np.save('./data/train_data.npy', train_data)
    # np.save('./data/train_label.npy', train_label)
    # print("Read training data and label ... ")
    # train_data = np.load('./data/train_data.npy')
    # train_label = np.load('./data/train_label.npy')

    train_data_test = train_data[int(len(train_data)*9/10):]
    train_label_test = train_label[int(len(train_label)*9/10):]

    train_data = train_data[:int(len(train_data)*9/10)]
    train_label = train_label[:int(len(train_label)*9/10)]

    ITERATION = 70
    BATCH_SIZE = 128

    train_data_gen = ImageDataGenerator(
                        rotation_range=30,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        zoom_range=[0.8, 1.2],
                        shear_range=0.2,
                        horizontal_flip=True)

    model = Sequential()
    model.add(Convolution2D(64, (3, 3), padding='same',
                     input_shape=(train_data.shape[1], train_data.shape[2], train_data.shape[3])))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(512, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Convolution2D(512, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.35))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=0.0005)
    model.compile(loss='categorical_crossentropy',
                  optimizer= optimizer,
                  metrics=['accuracy'])
    model.summary()
    # # early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    # train_history = model.fit(train_data, train_label, epochs=ITERATION, batch_size=BATCH_SIZE, verbose=1, validation_data=(train_data_test, train_label_test))

    train_history = model.fit_generator(
            train_data_gen.flow(train_data, train_label, batch_size=BATCH_SIZE),
            steps_per_epoch=5*len(train_data)//BATCH_SIZE,
            epochs=ITERATION,
            validation_data=(train_data_test, train_label_test)
            )

    score = model.evaluate(train_data_test, train_label_test)
    print("Loss: {}".format(score[0]))
    print("Accuracy: {}".format(score[1]))
    #
    y_pred = model.predict_classes(train_data_test)
    class_names = ['Angry', 'Digust', 'Fear', 'Happy', 'Sad', 'Suprise', 'Neutral']
    cnf_matrix = confusion_matrix(np.argmax(train_label_test,axis=1), y_pred)
    utils.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

    plot_model(model, to_file='cnn_model.png')
    model.save('./model_cnn_v9.h5')
    utils.show_train_history(train_history, 'acc', 'val_acc', 'acc.png')
    utils.show_train_history(train_history, 'loss', 'val_loss', 'loss.png')
    # del model
