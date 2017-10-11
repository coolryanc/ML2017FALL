import sys
import os
import json
import numpy as np
from numpy.linalg import inv
import dbHelper

def closeForm(traingData, predict):
    close_weights = np.matmul(np.matmul(inv(np.matmul(traingData.T,traingData)),traingData.T),predict)
    return close_weights

def scaleData(traingData):
    v_min = np.min(traingData, axis=0)
    v_max = np.max(traingData, axis=0)
    return (traingData - v_min) / (v_max - v_min)

def reshapeData(traingData, HOURS, traingFeature):
    feature_X = []
    predict_Y = []
    for index in range(len(traingData)-(HOURS+1)):
        getFeature = [1]
        predict_Y.append([traingData[index+HOURS][9]])
        for i in range(index, index+HOURS):
            for c in traingFeature:
                getFeature.append(traingData[i][c])
            getFeature.append(traingData[i][8]**2)
            getFeature.append(traingData[i][9]**2)
        feature_X.append(getFeature)
    return np.array(feature_X), np.array(predict_Y)

def calculateError(traingData, predict, weights, bias):
    return predict - (np.dot(traingData, weights) + bias)

def calculateLoss(traingData, predict, weights, bias, lamda):
    return np.sqrt(np.mean(calculateError(traingData, predict, weights, bias) ** 2) + lamda * np.sum(weights ** 2))

def predictPM_2_5(testingData, weights, HOURS, traingFeature):
    result = []
    for i in range(len(testingData)):
        result.append(np.dot(testingData[i], weights[1:]) + weights[0][0])
    return result

def main(argv):
    ITERATION = 50000
    LEARNING_RATE = 0.5
    DATA_CATEGORIES = 18
    HOURS = 9
    LAMDA = 0.0

    # traingFeature = [9]
    traingFeature = [7,8,9,10,14,15,16,17]
    # traingFeature = [0,7,8,9]
    # traingFeature = np.array(range(18))

    # getData = dbHelper.readTrainingData(DATA_CATEGORIES)
    # traingData, predict = reshapeData(getData[0:], HOURS, traingFeature)
    # traingData = scaleData(traingData)
    # validation, validationResult = reshapeData(getData[int(len(getData) * 9/10):len(getData)], HOURS, traingFeature)
    # validation = scaleData(validation)

    # close_weights = closeForm(traingData, predict)
    # close_loss = calculateLoss(traingData, predict, close_weights, 0.0, LAMDA)
    # print("Close form {}".format(close_loss))

    # np.save('./model/best_model.npy',close_weights)
    final_weights = np.load('./model/best_model.npy')

    writeText = "id,value\n"
    testingData = dbHelper.readTestingData(argv[1], DATA_CATEGORIES, traingFeature, HOURS, True)
    # testingData = scaleData(testingData)
    result = predictPM_2_5(testingData, final_weights, HOURS, traingFeature)
    for i in range(len(result)):
        writeText += "id_" + str(i) + "," + str(result[i][0]) + "\n"
    filename = argv[2]
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(writeText)


if __name__ == '__main__':
    main(sys.argv)
