#coding:utf-8

import numpy  # 感知机学习算法demo
from sklearn.preprocessing import StandardScaler
from sklearn import datasets#导入数据库
breast_cancer_data=datasets.load_breast_cancer()
features=breast_cancer_data.data#特征
targets=breast_cancer_data.target#类别

import sklearn.model_selection
X_train,X_test, y_train, y_test =sklearn.model_selection.train_test_split\
    (features,targets,test_size=0.1, random_state=0)
#将数据分为训练集和测试集

sc = StandardScaler()
sc.fit(features)
features= sc.transform(features)

for temp in targets:
    if targets[temp]==0:
        targets[temp]==-1
#将类别改为课上使用的-1

w = numpy.zeros(30)#设置w初始值200
b = 0#设置b初始值
text_num=int(input("请输入训练次数:"))
learn_rate=float(input("请输入学习率："))
for t in range(text_num):
    for e in range(len(X_train)):
        y=sum(w*X_train[e])+b
        if y*y_train[e]<=0:
            w=w+y_train[e]*X_train[e]*learn_rate
            b=b+y_train[e]*learn_rate
w_num=0#错误数据数量
for m in range(len(X_test)):
    y_2 = sum(w * X_test[m]) + b
    if y_2*y_test[m]<=0:
       w_num=1+w_num
print("测试集中正确率:%.2f%%" % ((1.0 - w_num / len(X_test)) * 100))

