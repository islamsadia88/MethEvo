import numpy as np
from tqdm import tqdm
import math

K = 100


def eucledian_distance(x, y):
    sum = 0
    for i in range(len(x)):
        now = (x[i] - y[i]) * (x[i] - y[i])
        sum += now
    return math.sqrt(sum)


def eucledian_distance_mat(x, y):
    x = np.asarray(x, np.float)
    y = np.asarray(y, np.float)
    now = x - y
    now = now.dot(now)
    return math.sqrt(np.sum(now))


def take_first(ele):
    return ele[0]


def knn_imbalance(x_p, x_n, y_n):
    print(len(x_p), len(x_n))
    formatted_n_x = []
    formatted_n_y = []
    for (neg_ind, x) in enumerate(tqdm(x_n)):
        now = []
        for i in x_p:
            now.append((eucledian_distance_mat(x, i), 1))
        for (neg_i, i) in enumerate(x_n):
            if neg_ind != neg_i:
                now.append((eucledian_distance_mat(x, i), 0))
        now.sort(key=take_first)

        for (ind, ele) in enumerate(now):
            if ind == K:
                break
            if ele[1] == 1:
                flg = 1
                break
        if flg == 0:
            formatted_n_x.append(x)
            formatted_n_y.append(y_n[neg_ind])
    print(len(x_p), len(formatted_n_x), len(formatted_n_y))
    return formatted_n_x, formatted_n_y


if __name__ == '__main__':
    npzfile = np.load('../Results_regeneration/Whole dataset/Our model (PSSM+SPD)/whole_data_features.npz', allow_pickle=True)
    X_p = npzfile['arr_0']
    Y_p = npzfile['arr_1']
    X_n = npzfile['arr_2']
    Y_n = npzfile['arr_3']

    print("The length")
    print(len(Y_p))
    print(len(Y_n))

    #print(X_p.shape, X_n.shape)
    #X_n, Y_n = knn_imbalance(X_p, X_n, Y_n)
    #np.savez('../data1/knn_features.npz', X_p, Y_p, X_n, Y_n)

