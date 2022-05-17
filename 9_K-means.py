from sklearn.datasets import load_digits
from sklearn import preprocessing
import copy
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn import manifold, datasets

data_M, labels = datasets.load_digits(return_X_y=True)
n_samples, n_features = data_M.shape

np.random.seed(0)

condition = 1
used = 1500
centers = []


# 求中心
def center(label, center):
    c0 = copy.deepcopy(center)

    for j1 in range(64):
        center[label][j1] = 0

    count = 0
    for i2 in range(used):
        if label == labels[i2]:
            count = count + 1
            for j2 in range(64):
                center[label][j2] = center[label][j2] + data_M[i2][j2]

    for k2 in range(64):
        if center[label][k2] != 0:
            center[label][k2] = center[label][k2] / count
    print(center[label])

    if c0[label] != center[label]:
        print(1)
        return 1

    print(0)
    return 0


def distance2(point, c):
    d = 0
    for j in range(64):
        d = d + (point[j]-c[j])*(point[j]-c[j])
    return d

# 迭代
def repeat(center):
    global labels
    for i in range(len(data_M)):
        d_min = 999999
        for j in range(len(center)):
            d = distance2(data_M[i], center[j])
            if d_min > d:
                d_min = d
                labels[i] = j

# 用层次聚类做预处理
def preprocess():
    global labels
    global data_M
    global centers
    digits = load_digits()
    print(digits.data.shape)

    data = np.array(digits.data[:used, :])
    print(data.shape)
    min_max_scaler = preprocessing.MinMaxScaler()
    data_M = min_max_scaler.fit_transform(data)
    print(data_M)
    print(data_M.shape)

    for i in range(used):
        labels[i] = -1

    for i in range(10):
        labels[i] = i

def nudge_images(X, y):
    shift = lambda x: ndimage.shift(x.reshape((8, 8)),0.3 * np.random.normal(size=2),mode='constant',).ravel()
    X = np.concatenate([X, np.apply_along_axis(shift, 1, X)])
    Y = np.concatenate([y, y], axis=0)
    return X, Y

X, y = nudge_images(data_M, labels)

def plot_clustering(X_red, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                 color=plt.cm.nipy_spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

preprocess()

centers = [0 for i in range(10)]
for i in range(10):
    centers[i] = [0 for j in range(64)]    # 用层次聚类做预处理
for i in range(10):
    center(i, centers)

while condition == 1:
    repeat(centers)

    temp = 0
    for i in range(10):
        temp = temp + center(i, centers)

    if temp == 0:
        condition = 0

X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(data_M)

plot_clustering(X_red, labels, "k-means")

plt.show()