import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
import parseData


if __name__ == '__main__':
    X_train, word_num, sequence_length, y_train = parseData.readTrainingData('./data/')
    print('\ntraining_data shape: {}'.format(X_train.shape))   # shape: (200000, 16, 50)
    print('training_label shape: {}'.format(y_train.shape)) # shape: (200000, 2)

    # fix random seed for reproducibility
    # load the dataset but only keep the top n words, zero the rest
    dataNums = word_num #X_train.shape[0]
    max_review_length = sequence_length
    # X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)

    # create the model
    embedding_vecor_length = 64
    EPOCHS = 7
    model = Sequential()
    model.add(Embedding(dataNums, embedding_vecor_length, input_length=max_review_length, trainable=True))
    # model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(512, batch_input_shape=(None, max_review_length, embedding_vecor_length), return_sequences=True)))
    # model.add(Bidirectional(LSTM(512)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(256)))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    earlystopping = EarlyStopping(monitor='val_loss', patience = 10)
    checkpoint = ModelCheckpoint(filepath='./model/best.h5', save_best_only=True, monitor='val_loss')
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=128, callbacks=[checkpoint],validation_split = 0.1)

    model.save('./model/rnn_model_1.h5')  # creates a HDF5 file 'my_model.h5'
    # del model
