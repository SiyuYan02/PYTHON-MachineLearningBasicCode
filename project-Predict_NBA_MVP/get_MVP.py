import pandas
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn import svm,neighbors,tree,linear_model,ensemble
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn import linear_model
from sklearn import ensemble

#数据处理

def get_data(data_train, data_test):
    data_np = np.asarray(data_train)

    train_label = data_np[:,1]
    train_data = data_np[:,2:]
    # 仅利用相关系数排名前五的特征作为预测变量,除去其余数据
    train_data = np.delete(train_data, [0, 1,  5, 6, 7, 8, 9, 10], axis=1)
    for i in range(data_np.shape[0]):
        train_data[i] = np.asarray(list(map(float,train_data[i, :])))

    #测试数据处理与训练数据相同
    data_np1 = np.asarray(data_test)
    player_name = data_np1[:,0]
    test_label = data_np1[:, 1]
    test_data = data_np1[:, 2:]
    test_data = np.delete(test_data,  [0, 1, 5  , 6, 7, 8, 9, 10], axis=1)
    for i in range(data_np1.shape[0]):
        test_data[i] = np.asarray(list(map(float, test_data[i, :])))

    return player_name, train_data, train_label, test_data, test_label


if __name__ == "__main__":
    data_train = pandas.read_csv("basketball_outstanding_player.csv")
    string = str.format("player_regular_season_2005.csv")
    data_test = pandas.read_csv(string)
    player_name, train_data, train_label, test_data, test_label = get_data(data_train, data_test)

    model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20) # 初始化随机树回归
    model_RandomForestRegressor.fit(train_data, train_label) # 训练随机树模型
    y_predict = model_RandomForestRegressor.predict(test_data) # 最终预测
    y_predict_temp = y_predict.tolist()
    max_number = y_predict_temp.index(max(y_predict))
    print(player_name[max_number])
