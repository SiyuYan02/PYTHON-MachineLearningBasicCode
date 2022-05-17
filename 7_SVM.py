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
c=[]
x_label=[]
#储存准确率
plt.figure(figsize=(120,8),dpi=80)
for i in range(1,10):
    c.append(1e-04*(10**i))
    x_label.append(i)
    clf = svm.SVC(kernel='rbf', gamma=0.1, decision_function_shape='ovo', C=1e-04*(10**i))
    #训练
    clf.fit(x_train, y_train)
    #测试的准确率
    y.append(clf.score(x_test, y_test))

plt.plot(x_label,y,'b',linewidth=2)
plt.bar(x_label,y,width=0.5,color='g')
plt.xticks(x_label,c,rotation=20)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签']
plt.xlabel("C取值")
plt.ylabel("测试准确率")
for i in range(0,9):
    plt.text(x_label[i], y[i] + 0.03, str('{:.2f}'.format(100 * y[i])) + '%', \
             fontsize=10, color="b", verticalalignment='center', horizontalalignment='center')
plt.show()

