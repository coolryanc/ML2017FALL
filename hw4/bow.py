import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Flatten, TimeDistributed
from keras.layers import LSTM, Bidirectional, LeakyReLU, BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
import parseData
numpy.random.seed(7)

if __name__ == '__main__':
    X_train, y_train, X_test, word_num, sequence_length  = parseData.bowSequence('./data/')
    print('\ntraining_data shape: {}'.format(X_train.shape))   # shape: (200000, 36)
    print('training_label shape: {}'.format(y_train.shape)) # shape: (200000, )

    WORDNUM = word_num
    max_review_length = sequence_length
    embedding_vecor_length = 100
    EPOCHS = 4
    BATCHSIZE = 64

    # create the model
    model = Sequential()
    # model.add(Embedding(WORDNUM, embedding_vecor_length, input_length=max_review_length, trainable=True))
    # model.add(Flatten())
    model.add(Dense(512, input_shape=(WORDNUM,), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(256, input_shape=(WORDNUM,), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    earlystopping = EarlyStopping(monitor='val_loss', patience = 10)
    checkpoint = ModelCheckpoint(filepath='./model/bowbest.h5', save_best_only=True, monitor='val_loss')
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCHSIZE, callbacks=[checkpoint],validation_split = 0.1)

    model.save('./model/bowModel.h5')  # creates a HDF5 file 'my_model.h5'
    # del model
