import sys, os
import numpy as np
import pandas as pd

# $3: provided train feature (X_train)
# $4: provided train label (Y_train)
# $5: provided test feature (X_test)
# $6: prediction.csv

def readTrainData():
    # traing Data
    X_train = pd.read_csv('./data/X_train').as_matrix() #shape: (32561, 106)
    Y_train = pd.read_csv('./data/Y_train').as_matrix() #shape: (32561, 1)
    Y_train = Y_train.reshape(Y_train.shape[0]) #shape: (32561,)

    X_train = np.concatenate((X_train[:,1:], X_train[:,[1,3,4,5]]**2, X_train[:,[1,3,4,5]]**3, np.log(X_train[:, [3]] + 1e-100)), axis=1)
    mean = np.mean(X_train, axis=0) #shape: (106,)
    std = np.std(X_train, axis=0) #shape: (106,)
    X_train = (X_train - mean) / (std + 1e-100)

    return X_train, Y_train, mean, std

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 0.00000000000001, 0.99999999999999)

def calculateError(X_train, Y_train, weights, bias):
    f = sigmoid(np.dot(X_train, weights) + bias)
    return Y_train - f, f

def calculateLoss(Y_train, f):
    return -np.mean(Y_train*np.log(f+1e-100) + (1-Y_train)*np.log(1-f+1e-100))

def calculateAccuracy(Y_train, f):
    f[f >= 0.5] = 1
    f[f < 0.5] = 0
    acc = Y_train - f #shape: (32561,)
    acc[acc == 0] = 2
    acc[acc != 2] = 0
    return np.sum(acc) * 50 / acc.shape[0]

def logisticRegression(LEARNING_RATE, ITERATION, X_train, Y_train):
    bias_descent = 0.0 # init bias
    weights_descent = np.ones(X_train.shape[1]) # init weights
    B_lr = 0.0 #Adagrad
    W_lr = np.zeros(X_train.shape[1]) #Adagrad

    for index in range(ITERATION):
        error, f = calculateError(X_train, Y_train, weights_descent, bias_descent)

        B_grad = -np.sum(error) * 1.0
        W_grad = -np.dot(X_train.T, error) # renew each weights

        B_lr += B_grad ** 2
        W_lr += W_grad ** 2

        bias_descent = bias_descent - LEARNING_RATE / np.sqrt(B_lr) * B_grad
        weights_descent = weights_descent - LEARNING_RATE / np.sqrt(W_lr) * W_grad
        current_loss = calculateLoss(Y_train, f)
        current_accuracy = calculateAccuracy(Y_train, f)

        print('\rIteration: {}, Accuracy: {}, Loss: {}'.format(str(index+1), current_accuracy, current_loss), end='' ,flush=True)
    print()
    return bias_descent, weights_descent

def main(argv):
    LEARNING_RATE = 0.5
    ITERATION = 5000
    X_train, Y_train, mean, std = readTrainData()
    final_bias, final_weights = logisticRegression(LEARNING_RATE, ITERATION, X_train, Y_train)
    writeText = "id,label\n"
    X_test = pd.read_csv('./data/X_test').as_matrix() #shape: (16281, 106)
    X_test = np.concatenate((X_test[:,1:], X_test[:,[1,3,4,5]]**2, X_test[:,[1,3,4,5]]**3, np.log(X_test[:, [3]] + 1e-100)), axis=1)
    X_test = (X_test - mean) / (std + 1e-100)

    predict_income = final_bias + np.dot(X_test, final_weights)

    for index, i in enumerate(predict_income):
        result = 0
        if i>=0:
            result = 1
        writeText += str(index+1) + "," + str(int(result)) + "\n"
    filename = './result/logistic.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(writeText)

if __name__ == '__main__':
    main(sys.argv)
    # readTrainData()
