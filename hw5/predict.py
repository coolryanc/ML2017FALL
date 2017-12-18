import numpy as np
import pandas as pd
from keras.models import load_model
import sys, os
import parseData
from cf_model import CF_Model
from deep_model import Deep_Model

def predict_rating(model, userid, movieid):
    return model.predict([np.array([userid-1]), np.array([movieid-1])])[0][0]

def main(argv):
    EMBEDDING_DIM = 200
    test_data, recommendations = parseData.read_test_data(argv[1])
    model = CF_Model(6040, 3952, EMBEDDING_DIM)
    model.load_weights('./model/model_unbias.h5')
    model1 = CF_Model(6040, 3952, EMBEDDING_DIM)
    model1.load_weights('./model/model_unbias1.h5')

    # DNN
    # model = Deep_Model(6040, 3952, EMBEDDING_DIM)
    # model.load_weights('./model/deep_model_unbias.h5')

    recommendations['Rating_1'] = test_data.apply(lambda x: predict_rating(model, x['UserID'], x['MovieID']), axis = 1)
    recommendations['Rating_2'] = test_data.apply(lambda x: predict_rating(model1, x['UserID'], x['MovieID']), axis = 1)
    recommendations['Rating'] = (recommendations['Rating_1'] + recommendations['Rating_2']) / 2
    filename = argv[2]
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    recommendations.to_csv(filename, index=False, columns=['TestDataID', 'Rating'])
    print(recommendations)

if __name__ == '__main__':
    main(sys.argv)
