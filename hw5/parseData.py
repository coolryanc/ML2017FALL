import numpy as np
import pandas as pd
import sys, os

def read_data(dataPath):
    # MAX_USERID: 6040
    # MAX_MOVIEID: 3952
    # UserID: [ 795  795  795 ..., 2682 2682 2682], shape = (899873,)
    # MovieID: [1192  660  913 ..., 1093  561 1096], shape = (899873,)
    # Rating: [5 3 3 ..., 5 5 4], shape = (899873,)
    ratings = pd.read_csv(dataPath,
                          usecols=['UserID', 'MovieID', 'Rating'])
    MAX_USERID = ratings['UserID'].drop_duplicates().max()
    MAX_MOVIEID = ratings['MovieID'].drop_duplicates().max()
    ratings['User_emb_id'] = ratings['UserID'] - 1
    ratings['Movie_emb_id'] = ratings['MovieID'] - 1

    UserID = ratings['User_emb_id'].values
    MovieID = ratings['Movie_emb_id'].values
    Rating = ratings['Rating'].values

    return MAX_USERID, MAX_MOVIEID, UserID, MovieID, Rating

def read_test_data(dataPath):
    test_data = pd.read_csv(dataPath, usecols = ['UserID', 'MovieID'])
    recommendations = pd.read_csv(dataPath, usecols = ['TestDataID'])
    return test_data, recommendations
