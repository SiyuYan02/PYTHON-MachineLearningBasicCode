from sklearn import model_selection, linear_model, tree
import random
import math
import pydotplus
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import pylab as mpl

#去掉警告信息
warnings.filterwarnings('ignore')
#处理中文乱码
plt.rcParams['font.sans-serif']=['SimHei']    # 用来设置字体样式以正常显示中文标签
plt.rcParams['axes.unicode_minus']=False    # 默认是使用Unicode负号，设置正常显示字符，如正常显示负号



def pretreatment(Mstat,Ostat,Tstat):

    # 去掉无关数据
    new_M = Mstat.drop(['Rk', 'Arena', 'Attend.', 'Attend./G'], axis=1)
    new_O = Ostat.drop(['Rk', 'G', 'MP'], axis=1)
    new_T = Tstat.drop(['Rk', 'G', 'MP'], axis=1)


    new_M['Team'] = new_M.apply(lambda x: x, axis=1)
    new_O['Team'] = new_O.apply(lambda x: x, axis=1)
    new_T['Team'] = new_T.apply(lambda x: x, axis=1)

    # 表格连接
    team_stats = pd.merge(new_M, new_O, how='left', on='Team')
    team_stats = pd.merge(team_stats, new_T, how='left', on='Team')
    return team_stats.set_index('Team', inplace=False, drop=True)


# 计算客场胜或是主场胜
def formed_data(res):
    res_new = pd.DataFrame(columns=['WTeam', 'LTeam', 'WLoc'])
    for index, row in res.iterrows():
        if row['PTS'] > row['PTS.1']:
            res_new.loc[index, 'WTeam'] = row['Visitor/Neutral']
            res_new.loc[index, 'LTeam'] = row['Home/Neutral']
            res_new.loc[index, 'WLoc'] = 'V'
        else:
            res_new.loc[index, 'WTeam'] = row['Home/Neutral']
            res_new.loc[index, 'LTeam'] = row['Visitor/Neutral']
            res_new.loc[index, 'WLoc'] = 'H'
    return res_new

# 建立数据集
def build_dataSet(team_stats, all_data):

    X = []
    y = []
    skip = 0

    for index, row in all_data.iterrows():
        Wteam = row['WTeam']
        Lteam = row['LTeam']

        try:
            team1_elo = team_elos[Wteam]
        except:
            # 赋初值
            team1_elo = 1000

        try:
            team2_elo = team_elos[Lteam]
        except:
            # 赋初值
            team2_elo = 1000
        # 给主场比赛的队伍加上200
        if row['WLoc'] == 'H':
            team1_elo += 200
        else:
            team2_elo += 200

        team1_features = [team1_elo]
        team2_features = [team2_elo]

        # 添加其余球队基础数据
        for key, value in team_stats.loc[Wteam].iteritems():
            team1_features.append(value)
        for key, value in team_stats.loc[Lteam].iteritems():
            team2_features.append(value)
        # 将两支队伍的特征值随机的分配在每场比赛数据的左右两侧
        # 并将对应的0/1赋给y
        # 主队赢y为0，客队赢y为1
        if random.random() > 0.5:
            X.append(team1_features + team2_features)
            y.append(0)
        else:
            X.append(team2_features + team1_features)
            y.append(1)
        if skip == 0:
            skip = 1

    return np.nan_to_num(X), y

def get_data(data_list):

    team_stats = {}
    X = []
    Y = []

    #预处理数据
    Mstat_train = pd.read_csv(data_list[0], header=1)
    Ostat_train = pd.read_csv(data_list[1])
    Tstat_train = pd.read_csv(data_list[2])
    team_stats = pretreatment(Mstat_train, Ostat_train, Tstat_train)

    result_data_train = pd.read_csv(data_list[3])
    res_new_train = formed_data(result_data_train)

    X, Y = build_dataSet(team_stats, res_new_train)
    return X, Y

if __name__ == '__main__':
    team_elos = {}

    # 读入数据
    data_train_list = ['data/mcs_train.txt', 'data/ops_train.txt', 'data/tps_train.txt', 'data/res_train.txt']
    data_test_list = ['data/mcs_test.txt', 'data/ops_test.txt', 'data/tps_test.txt', 'data/res_test.txt']

    # 获取训练数据
    team_data_tra = {}
    # 预处理数据
    Mstat_train = pd.read_csv(data_train_list[0], header=1)
    Ostat_train = pd.read_csv(data_train_list[1])
    Tstat_train = pd.read_csv(data_train_list[2])
    team_data_tra = pretreatment(Mstat_train, Ostat_train, Tstat_train)

    result_data_train = pd.read_csv(data_train_list[3])
    res_new_train = formed_data(result_data_train)

    X_train, y_train = build_dataSet(team_data_tra, res_new_train)
    # 获取测试数据
    team_data_te = {}
    # 预处理数据
    Mstat_test = pd.read_csv(data_test_list[0], header=1)
    Ostat_test = pd.read_csv(data_test_list[1])
    Tstat_test = pd.read_csv(data_test_list[2])
    team_data_te = pretreatment(Mstat_test, Ostat_test, Tstat_test)

    result_data_test = pd.read_csv(data_train_list[3])
    res_new_test = formed_data(result_data_test)

    X_test, y_test = build_dataSet(team_data_te, res_new_test)

    # 训练模型
    model = tree.DecisionTreeClassifier(criterion="entropy", max_depth=6, min_samples_leaf=6)
    model.fit(X_train, y_train)


    # 利用10折交叉验证计算训练正确率
    x = list(range(10))
    score_acc = model_selection.cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)
    score_ave = model_selection.cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1).mean()
    # 利用测试值计算正确率
    test_acc = model_selection.cross_val_score(model, X_test, y_test, cv=10, scoring='accuracy', n_jobs=-1).mean()

    # 结果可视化
    plt.scatter(x, score_acc, marker='v')
    plt.axhline(y=score_ave, ls=":", c="black")
    plt.plot(score_acc)
    plt.title('DecisionTreeClassifier')
    plt.legend(['average', 'each time'])
    plt.suptitle("预测正确率", fontsize=16)
    plt.show()

