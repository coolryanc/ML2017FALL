import sys
import json
import numpy as np
# import matplotlib.pyplot as plt

def readTrainingData(DATA_CATEGORIES):
    tmpData = [] #[hour][date]
    finalData = []
    print("Start parsing Training Data")
    # np.delete(data, index, (row=0, col =1))
    data = np.delete(np.genfromtxt('train.csv', delimiter=','),0,0) #del 日期、測站...
    for index in range(3): #del date, location, category
        data = np.delete(data,0,1)
    data = data.T #transpose data matrix
    for hour in range(24):
        data[hour][np.isnan(data[hour])] = 0
        tmpData.append(np.split(data[hour], len(data[hour])/DATA_CATEGORIES)) #split to 240 days
    tmpData = np.array(tmpData)
    for date in range(0,len(tmpData[0])):
        for hour in range(24):
            finalData.append(tmpData[hour][date])
    finalData = np.array(finalData)
    print("Finish reading Training Data")
    return finalData

def readTestingData(path, DATA_CATEGORIES, traingFeature, HOURS, isSquareFeature):
    tmpData = [] #[hour][date]
    finalData = []
    data = np.genfromtxt(path, delimiter=',')
    for i in range(2):
        data = np.delete(data,0,1)
    data = data.T
    for hour in range(len(data)):
        data[hour][np.isnan(data[hour])] = 0
        tmpData.append(np.split(data[hour], len(data[hour])/DATA_CATEGORIES))
    tmpData = np.array(tmpData)
    feature_X = []
    for i in range(len(tmpData[0])):
        getFeature = []
        for h in range(HOURS):
            for c in traingFeature:
                getFeature.append(tmpData[h][i][c])
            if isSquareFeature:
                getFeature.append(tmpData[h][i][8]**2)
                getFeature.append(tmpData[h][i][9]**2)
        feature_X.append(getFeature)
    return np.array(feature_X)
if __name__ == '__main__':
    pass
