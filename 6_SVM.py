from sklearn import svm  # svm函数需要的
from sklearn import datasets
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt

iris=datasets.load_iris()
data=iris.data
target=iris.target
x_train, x_test, y_train, y_test = model_selection.train_test_split(data, target, random_state=1, test_size=0.3)

clf = svm.SVC(kernel='rbf', gamma=0.1, decision_function_shape='ovo', C=0.8)

#训练
clf.fit(x_train, y_train)
#打印训练的准确率
print('训练的准确率',clf.score(x_train, y_train))

#打印测试的准确率
print('测试的准确率',clf.score(x_test, y_test))

