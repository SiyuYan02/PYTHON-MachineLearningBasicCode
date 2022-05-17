from sklearn import datasets
import numpy as np
import random
import sklearn.model_selection
from numpy.matlib import repmat
import copy
import math

#download data
iris=datasets.load_iris()
data=iris.data
target=iris.target

#将数据分为训练集和测试集
X_train,X_test, y_train, y_test =sklearn.model_selection.train_test_split\
    (data,target,test_size=0.1, random_state=0)
len_=len(X_train)
len_n=len(X_test)

def one_vs_test_1(target):
    for i in range(len_):
        if target[i]==0:
            target[i]=1
        else:
            target[i] = 0
    return target

def one_vs_test_2(target):
    for i in range(len_):
        if target[i]==1:
            target[i]=1
        else:
            target[i] == 0
    return target

def one_vs_test_3(target):
    for i in range(len_):
        if target[i]==2:
            target[i]=1
        else:
            target[i] = 0

    return target

#sigmod
def sigmoid(w,x):  #sigmoid函数
    wx = np.dot(w, x)  # 计算点乘
    return math.exp(wx) / (1 + math.exp(wx))

#梯度下降法
def gradAscent(X,Y):
    n_sample, n = X.shape
    W = np.ones((n, 1))
    alpha = 0.001
    Iter = 5000
    # 对数似然函数
    for i in range(Iter):
        # 最大似然估计
        Loss = Y * (X.dot(W)) - np.log(1 + np.exp(X.dot(W)))
        # 计算梯度
        grad = (Y * X) - (1 / (1 + np.exp(X.dot(W)))) * np.exp(X.dot(W)) * X
        grad = grad.mean(0)
        grad = grad.reshape([n, 1])
        # 梯度上升
        W = W + alpha * grad
    return W

#计算极大似然
def col(X,W):
    arr = np.array(W)
    # 这里可以三种方法达到转置的目的
    # 第一种方法
    q=arr.T
    print(q)
    h = sigmoid(q , X)
    return h

#测试
def Test(w, testx, testy):
    l = len(testx)
    ac = 0
    for i in range(l):
        p1 = []
        for j in range(len(w)):  # 由于采用了one vs rest方法，需要遍历所有的w
            p1.append(col(w[j], testx[i]))
        y = p1.index(max(p1))  # 选择其中最大的值，将它的分类结果当作最后的预测结果
        if y == testy[i]:  # 比较
            ac += 1
    return ac / l

#学习
y_train_1=y_train
y_train_2 = copy.deepcopy(y_train)
y_train_3 = copy.deepcopy(y_train)
Y_train_1=one_vs_test_1(y_train_1)
Y_train_2=one_vs_test_2(y_train_2)
Y_train_3=one_vs_test_3(y_train_3)

Y_train_1=Y_train_1.reshape((Y_train_1.shape[0],1))
Y_train_2=Y_train_2.reshape((Y_train_2.shape[0],1))
Y_train_3=Y_train_3.reshape((Y_train_3.shape[0],1))

w0 = gradAscent(X_train,Y_train_1)
w1 = gradAscent(X_train,Y_train_2)
w2 = gradAscent(X_train,Y_train_3)
ww = [w0, w1, w2]
print("Accuracy: ", Test(ww, X_test, y_test))
