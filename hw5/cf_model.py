import argparse
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda, Embedding, Reshape, Merge, Dropout, Dense, Flatten, Dot
from keras.models import Sequential, Model
from keras import backend as K
from keras.layers.merge import concatenate, dot, add
from keras.regularizers import l2
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.initializers import Zeros
# from keras.utils import plot_model
import parseData


def CF_Model(MAX_USERID, MAX_MOVIEID, EMBEDDING_DIM):
    U_input = Input(shape = (1,))
    U = Embedding(MAX_USERID, EMBEDDING_DIM, embeddings_regularizer=l2(0.00001))(U_input)
    U = Reshape((EMBEDDING_DIM,))(U)
    U = Dropout(0.2)(U)

    V_input = Input(shape = (1,))
    V = Embedding(MAX_MOVIEID, EMBEDDING_DIM, embeddings_regularizer=l2(0.00001))(V_input)
    V = Reshape((EMBEDDING_DIM,))(V)
    V = Dropout(0.2)(V)

    U_bias = Embedding(MAX_USERID, 1, embeddings_regularizer=l2(0.00001))(U_input)
    U_bias = Reshape((1,))(U_bias)

    V_bias = Embedding(MAX_USERID, 1, embeddings_regularizer=l2(0.00001))(V_input)
    V_bias = Reshape((1,))(V_bias)

    out = dot([U,V], -1)
    out = add([out, U_bias, V_bias])

    out = Lambda(lambda x: x + K.constant(3.5817,  dtype=K.floatx()))(out)
    model = Model(inputs=[U_input, V_input], outputs=out)
    return model

def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1., 5.)
    return K.sqrt(K.mean(K.square((y_true - y_pred))))

def main():
    EMBEDDING_DIM = 512
    MAX_USERID, MAX_MOVIEID, UserID, MovieID, Rating = parseData.read_data()

    # mean = np.mean(Rating, axis=0)
    # std = np.std(Rating, axis=0)
    # Rating = (Rating - mean) / (std + 1e-100)

    model = CF_Model(MAX_USERID, MAX_MOVIEID, EMBEDDING_DIM)
    # plot_model(model, to_file='CFModel.png', show_shapes=True)
    model.summary()
    model.compile(loss='mse', optimizer='adamax', metrics=[rmse])
    # callbacks = [EarlyStopping('val_rmse', patience=2), ModelCheckpoint('./model/best.h5', save_best_only=True)]
    history = model.fit([UserID, MovieID], Rating, epochs=20, verbose=1, validation_split=.1, batch_size=512)
    model.save('./model/model_unbias_p1.h5')

if __name__ == '__main__':
    main()
