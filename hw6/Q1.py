from skimage import transform
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import sys, os

def load_img(path):
    matrix = []
    for i in range(415):
        img = io.imread(path + '/' + str(i) + '.jpg')
        img = transform.resize(img, (600, 600))
        img = img.flatten()
        matrix.append(img)
    matrix = np.array(matrix)
    return matrix

def draw_img(picture, path):
    picture = picture.reshape((600, 600, 3))
    picture -= np.min(picture)
    picture /= np.max(picture)
    picture = (picture * 255).astype(np.uint8)
    plt.imsave(path, picture)

if __name__ == '__main__':
    argv = sys.argv
    img_path = './img'
    eigenface_path = './img/eigenfaces/'
    face_path = './img/face'
    draw = [29, 92, 102, 397]
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(eigenface_path):
        os.makedirs(eigenface_path)

    X_ori = load_img(argv[1])
    all_mean = np.mean(X_ori, axis=0)
    all_mean = all_mean.reshape((600, 600, 3))
    plt.imsave('./img/all_mean.png', all_mean)
    X_mean = np.mean(X_ori, axis=1)
    X = X_ori.T
    U, s, V = np.linalg.svd(X - X_mean, full_matrices=False)

    eigenvalue_sum = np.sum(s)
    # print(s[:4] / eigenvalue_sum)

    for i in range(4):
        draw_img(U.T[i], eigenface_path + str(i) + '.jpg')

    for i in draw:
        weight = np.dot(X_ori[i], U[:,:4])
        img = np.matmul(U[:,:4], weight)
        img = img + X_mean[i]
        draw_img(img, face_path + str(i) + '.jpg')

    draw = int(argv[2].split('.')[0])
    weight = np.dot(X_ori[draw], U[:,:4])
    img = np.matmul(U[:,:4], weight)
    img = img + X_mean[draw]
    draw_img(img, 'reconstruction.jpg')
