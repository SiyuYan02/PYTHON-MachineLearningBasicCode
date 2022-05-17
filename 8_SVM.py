from sklearn import svm  # svm函数需要的
from sklearn import datasets
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
import math

iris=datasets.load_iris()
data=iris.data
target=iris.target

x_train, x_test, y_train, y_test = model_selection.train_test_split(data, target, test_size=0.3)
y=[]
x_label=[]
#储存准确率
plt.figure(figsize=(120,8),dpi=80)

clf_rbf = svm.SVC(kernel='rbf', gamma=0.1, decision_function_shape='ovo')#高斯
clf_poly = svm.SVC(kernel='poly', gamma=0.1, decision_function_shape='ovo')#多项式核函数
clf_linear = svm.SVC(kernel='linear', gamma=0.1, decision_function_shape='ovo')#线性核函数
clf_sigmoid = svm.SVC(kernel='sigmoid', gamma=0.1, decision_function_shape='ovo')#核函数
clf_precomputed = svm.SVC(kernel='precomputed', gamma=0.1, decision_function_shape='ovo')#核矩阵
#训练
clf_rbf.fit(x_train, y_train)
clf_poly.fit(x_train, y_train)
clf_linear.fit(x_train, y_train)
clf_sigmoid.fit(x_train, y_train)
#测试的准确率
y.append(clf_rbf.score(x_test, y_test))
y.append(clf_poly.score(x_test, y_test))
y.append(clf_linear.score(x_test, y_test))
y.append(clf_sigmoid.score(x_test, y_test))

x_label=['rbf','poly','linear','sigmoid']
plt.bar(x_label,y,width=0.5,color='g')
plt.xticks(x_label,rotation=20)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签']
plt.xlabel("核")
plt.ylabel("测试准确率")
for i in range(0,4):
    plt.text(x_label[i], y[i] + 0.03, str('{:.2f}'.format(100 * y[i])) + '%', \
             fontsize=15, color="r", verticalalignment='center', horizontalalignment='center')
plt.show()

