import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.model_selection import KFold
import math

class MyBayes:
    def __init__(self):
        self.model = None

    # 数学期望,踩坑，mean函数里不能加self,因为mean函数是静态方法
    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    def stdev(self, x):
        avg = self.mean(x)
        return math.sqrt(sum([math.pow(i - avg, 2) for i in x]) / len(x))
    # 高斯概密度函数
    def gaussian_probability(self, x, mean, stdev):
        ex = math.exp(-(pow(x - mean, 2) / (2 * pow(stdev, 2))))
        return (1 / math.sqrt(2 * math.pi * pow(stdev, 2))) * ex

    # 处理 X_train
    def summarize(self, train_data):
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]

        return summaries
    # 分别求出数学期望和标准差
    def fit(self, x, y):
        # 利用集合不重复的特点，求出y可能的取值
        labels = list(set(y))
        data = {label: [] for label in labels}
        for f, label in zip(x, y):
            data[label].append(f)
        #print(data.items())
        self.model = {label: self.summarize(value) for label, value in data.items()}
        return "ok"

    # 计算概率
    def calculate_probabilities(self, input_data):
        # summaries:{0.0: [(5.0, 0.37),(3.42, 0.40)], 1.0: [(5.8, 0.449),(2.7, 0.27)]}
        # input_data:[1.1, 2.2]
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(input_data[i], mean, stdev)

        return probabilities

    # 类别
    def predict(self, x_test):
        # 将预测数据在所有类别中的概率进行排序，并取概率最高的类别
        label = sorted(self.calculate_probabilities(x_test).items(), key=lambda x: x[-1])[-1][0]
        return label

    # 准确率
    def score(self, x_test, y_test):
        right = 0
        for x, y in zip(x_test, y_test):
            if self.predict(x) == y:
                right += 1
        a=right / float(len(x_test))
        return a


if __name__ == '__main__':
    # data
    def create_data():
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['label'] = iris.target
        df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
        data = np.array(df.iloc[:150, :])
        return data[:, :-1], data[:, -1]

    data, target = create_data()
    Train_X = []
    Test_X = []
    Train_Y = []
    Test_Y = []
    #kflod
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(data):
        for _ in train_index:
            Train_X.append(data[_])
            Train_Y.append(target[_])
        for _ in test_index:
            Test_X.append(data[_])
            Test_Y.append(target[_])
    ac = []
    for _ in range(5):
        model = MyBayes()
        model.fit(Train_X[_ * 120:(_ + 1) * 120], Train_Y[_ * 120:(_ + 1) * 120])
        ac.append(model.score(Test_X[_ * 30:(_ + 1) * 30], Test_Y[_ * 30:(_ + 1) * 30]))
    print(ac)
    s = 0
    for _ in range(5):  # 计算平均值
        s += ac[_]
    s /= 5
    print(s)
