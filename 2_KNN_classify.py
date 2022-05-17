from numpy import *
import operator

from sklearn import datasets#导入数据库
breast_cancer_data=datasets.load_breast_cancer()
features=breast_cancer_data.data#特征
targets=breast_cancer_data.target#类别

import sklearn.model_selection

X_train,X_test, y_train, y_test =sklearn.model_selection.train_test_split\
    (features,targets,test_size=0.1, random_state=0)
#将数据分为训练集和测试集

def classify_KNN(test,x_train,y_train,k):
    # numpy函数shape[0]返回dataSet的行数
    totalsSize = x_train.shape[0]
    # 将测试数据变成和训练数据一样的矩阵
    tests = tile(test, (totalsSize, 1)) - x_train
    # 二维特征相减后平方
    sqDiffMat = tests ** 2
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 返回distances中元素从小到大排序后的索引值
    sortedDistIndices = sqDistances.argsort()
    class_count = {}  # 创建一个字典
    for i in range(k):
      vote_i_label = y_train[ sortedDistIndices[i]]
      class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


testData=[]
testData=X_test

w_num=0#错误数据数量
#挨个测试然后每次测是否正确5
k=0
for y in range(20):
  k=k+1
  p=0
  w_num=0
  for test in testData:
        result=classify_KNN(test,X_train,y_train,k)
        if result==y_test[p]:
          w_num=1+w_num
        p=p+1
  print(k, ':',"测试集中正确率:%.2f%%" % ((w_num / len(X_test)) * 100))


