import argparse
import numpy as np
from keras.layers import Input, Lambda, Embedding, Reshape, Merge, Dropout, Dense, Flatten, Dot
from keras.models import Sequential, Model
from keras import backend as K
from keras.layers.merge import concatenate, dot, add
from keras.regularizers import l2
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
import parseData

def Deep_Model(MAX_USERID, MAX_MOVIEID, EMBEDDING_DIM):
    U_input = Input(shape=(1,))
    u = Embedding(MAX_USERID, EMBEDDING_DIM)(U_input)
    u = Flatten()(u)

    M_input = Input(shape=(1,))
    m = Embedding(MAX_MOVIEID, EMBEDDING_DIM)(M_input)
    m = Flatten()(m)

    out = concatenate([u, m])
    out = Dropout(0.1)(out)
    out = Dense(256, activation='relu')(out)
    out = Dropout(0.1)(out)
    out = Dense(128, activation='relu')(out)
    out = Dropout(0.1)(out)
    out = Dense(64, activation='relu')(out)
    out = Dropout(0.15)(out)
    out = Dense(EMBEDDING_DIM, activation='relu')(out)
    out = Dropout(0.2)(out)
    out = Dense(1, activation='relu')(out)

    model = Model(inputs=[U_input, M_input], outputs=out)
    return model


def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1., 5.)
    return K.sqrt(K.mean(K.square((y_true - y_pred))))

def main():
    EMBEDDING_DIM = 200
    MAX_USERID, MAX_MOVIEID, UserID, MovieID, Rating = parseData.read_data()

    # mean = np.mean(Rating, axis=0)
    # std = np.std(Rating, axis=0)
    # Rating = (Rating - mean) / (std + 1e-100)

    model = Deep_Model(MAX_USERID, MAX_MOVIEID, EMBEDDING_DIM)
    plot_model(model, to_file='DeepModel.png', show_shapes=True)
    # model.summary()
    model.compile(loss='mse', optimizer='adamax', metrics=[rmse])
    # callbacks = [EarlyStopping('val_rmse', patience=2), ModelCheckpoint('./model/deep_best.h5', save_best_only=True)]
    history = model.fit([UserID, MovieID], Rating, epochs=30, verbose=1, validation_split=.1, batch_size=512)
    model.save('./model/deep_model_unbias.h5')

if __name__ == '__main__':
    main()
