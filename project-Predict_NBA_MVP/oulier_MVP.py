import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import xlwt


#导入数据
player = pd.read_csv("player_regular_season.csv")
player = np.array(player)

player_MVP_candidate = pd.read_csv("data/basketball_outstanding_player.csv")
player_MVP_candidate = np.array(player_MVP_candidate)


#储存年份数据
x_year = []
x_year_ = []

#预处理数据
def pretreatment():

    for i in range(len(player)):
        # 一个球员一般一场30多分钟,一共82场,去除上场时间过少的球员
        if player[i][7] < 1000:
            x_year.append(i)
        # 将球员的Fristname和lastname合并
        player[i][2] += " "
        player[i][2] += player[i][3]
        # 将命中次数转化为命中率
        if player[i][21] != 0:
            player[i][22] /= player[i][21]
        if player[i][19] != 0:
            player[i][20] /= player[i][19]
        if player[i][17] != 0:
            player[i][18] /= player[i][17]
        # 将绝对数据转换为相对的场均数据
        for j in range(8, 15):
            player[i][j] /= player[i][6]

    # 删除掉处理后的无用数据
    y = [0, 3, 4, 5, 17, 19, 21]
    pre_player = np.delete(player, y, axis=1)
    pre_player = np.delete(pre_player, x_year, axis=0)
    return pre_player

#获取不同年份的数据
def read_player_year(year,player):

    for i in range(len(player)):
        if player[i][0] != year:
            x_year_.append(i)

    new_player = np.delete(player, [0], axis=1)
    #不同年份的数据
    player_year = np.delete(new_player, x_year_, axis=0)
    #比赛数据
    player_year_stat = np.delete(player_year, [0], axis=1)
    return player_year, player_year_stat

#写入excel中
def data_write(file_path, datas):
    f = xlwt.Workbook()
    sheet1 = xlwt.Workbook(encoding='utf-8',style_compression=0)
    sheet = sheet1.add_sheet('player', cell_overwrite_ok=True)
    col = ('Player', 'Share', 'Age', 'G', 'MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'WS', 'WS/48')
    for i in range(0, 15):
        sheet.write(0, 'player', col[i])
    for i in range(len(datas)):
        data = datas[i]
        for j in range(0, 15):
            sheet.write(i + 1, j, data[j])

    savepath = 'C:/Users/yan/PycharmProjects/pythonProject/data'
    book.save(savepath)


if __name__ == "__main__":

    #预处理数据
    pre_player=pretreatment()
    year = 2004
    # 获取2004年的常规赛球员数据
    player_year, player_year_stat = read_player_year(year,pre_player)
    # 初始化孤立森林,选择paper里推荐的参数取值
    M = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(1/ len(player_year_stat)),max_features=1.0)
    # 训练模型
    M.fit(player_year_stat)

    model_pre = M.predict(player_year_stat)
    # test_res = M.decision_function(player_year_stat)

    # 显示outliers球员
    outstanding_player = []

    for i in range(len(model_pre)):
        if model_pre[i] == -1:
            outstanding_player.append(i)

    for i in outstanding_player:
        print(player_year[i][0])
        # data_write("player_regular_season_2014.csv",player_year[i])





