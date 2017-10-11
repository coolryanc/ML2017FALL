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
        getFeature = []
        predict_Y.append([traingData[index+HOURS][9]])
        for i in range(index, index+HOURS):
            for c in traingFeature:
                getFeature.append(traingData[i][c])
        feature_X.append(getFeature)
    return np.array(feature_X), np.array(predict_Y)

def calculateError(traingData, predict, weights, bias):
    return predict - (np.dot(traingData, weights) + bias)

def calculateLoss(traingData, predict, weights, bias, lamda):
    return np.sqrt(np.mean(calculateError(traingData, predict, weights, bias) ** 2) + lamda * np.sum(weights ** 2))

def gradientDescent(traingData, predict, LEARNING_RATE, ITERATION, HOURS, LAMDA):
    # traingData [[72],[72],...]
    print("Start running gradient descent")
    bias_descent = 0.0
    weights_descent = np.ones((len(traingData[0]), 1))
    B_lr = 0.0  #Adagrad
    W_lr = np.zeros((len(traingData[0]), 1)) #Adagrad
    N = len(traingData)
    for index in range(ITERATION):
        error = calculateError(traingData, predict, weights_descent, bias_descent)
        B_grad = -np.sum(error) * 1.0 / N
        W_grad = -np.dot(traingData.T, error) / N # renew each weights

        B_lr += B_grad ** 2
        W_lr += W_grad ** 2

        bias_descent = bias_descent - LEARNING_RATE / np.sqrt(B_lr) * B_grad
        weights_descent = weights_descent * (1 - (LEARNING_RATE / np.sqrt(W_lr)) * LAMDA) - LEARNING_RATE / np.sqrt(W_lr) * W_grad
        current_loss = calculateLoss(traingData, predict, weights_descent, bias_descent, LAMDA)
        print('\rIteration: {}, Loss: {}'.format(str(index+1), current_loss), end='' ,flush=True)
    print()
    return np.concatenate((np.array([[bias_descent]]), weights_descent))


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
    traingFeature = [2,5,7,8,9,14,15,16,17]
    # traingFeature = [7,8,9,10,14,15,16,17]
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

    # final_weights = gradientDescent(traingData, predict, LEARNING_RATE, ITERATION, HOURS, LAMDA)
    # validation_Loss = calculateLoss(validation, validationResult, final_weights[1:], final_weights[0][0], LAMDA)
    # print("Validation Loss {}".format(validation_Loss))

    # np.save('./model/model.npy',final_weights)
    final_weights = np.load('./model/model.npy')

    writeText = "id,value\n"
    testingData = dbHelper.readTestingData(argv[1], DATA_CATEGORIES, traingFeature, HOURS, False)
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
