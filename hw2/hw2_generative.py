import sys, os
import numpy as np
import pandas as pd

# $3: provided train feature (X_train)
# $4: provided train label (Y_train)
# $5: provided test feature (X_test)
# $6: prediction.csv

def gaussianDistribution(X_test, u1, u2, sameCov, N1, N2):
    sameCov_inv = np.linalg.inv(sameCov) #shape: (106, 106)
    w = np.dot((u1-u2), sameCov_inv)
    x = X_test.T #shape: (106, 16281)
    b = (-0.5) * np.dot(np.dot([u1], sameCov_inv), u1) + (0.5) * np.dot(np.dot([u2], sameCov_inv), u2) + np.log(float(N1)/N2)
    result = sigmoid(np.dot(w,x) + b)
    return np.around(result)

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 0.00000000000001, 0.99999999999999)

def readTrainData(X_train, Y_train):
    # traing Data
    X_train = pd.read_csv(X_train).as_matrix() #shape: (32561, 106)
    Y_train = pd.read_csv(Y_train).as_matrix() #shape: (32561, 1)
    Y_train = Y_train.reshape(Y_train.shape[0]) #shape: (32561,)

    label_one_picker = (Y_train == 1) #shape: (32561,)
    label_zero_picker = (Y_train == 0) #shape: (32561,)
    one_train = X_train[label_one_picker, :] # select label 1 to List, get shape: (7841, 106)
    zero_train = X_train[label_zero_picker, :]

    return one_train, zero_train

def getCov(u, c):
    sigma = np.zeros((106,106))
    for i in c:
        sigma += np.dot(np.transpose([i - u]), [(i - u)])
    sigma /= c.shape[0]
    return sigma


def main(argv):
    class_one, class_zero = readTrainData(argv[1], argv[2])

    u1 = np.mean(class_one, axis=0)
    # cov1 = np.dot((class_one - u1).T, class_one - u1) / class_one.shape[0]
    u2 = np.mean(class_zero, axis=0)
    # cov2 = np.dot((class_zero - u2).T, class_zero - u2) / class_zero.shape[0]
    cov1 = getCov(u1, class_one)
    cov2 = getCov(u2, class_zero)

    prob_one = class_one.shape[0] / (class_one.shape[0] + class_zero.shape[0])
    prob_zero = 1 - prob_one
    sameCov = prob_one * cov1 + prob_zero * cov2

    writeText = "id,label\n"
    X_test = pd.read_csv(argv[3]).as_matrix() #shape: (16281, 106)

    predict_income = gaussianDistribution(X_test, u1, u2, sameCov, class_one.shape[0], class_zero.shape[0])

    for index, i in enumerate(predict_income):
        writeText += str(index+1) + "," + str(int(i)) + "\n"
    filename = argv[4]
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(writeText)

if __name__ == '__main__':
    main(sys.argv)
    # readTrainData()
