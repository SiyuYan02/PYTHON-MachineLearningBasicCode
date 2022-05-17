import pandas as pd
import numpy as np
import math
import operator
from sklearn.model_selection import KFold

#导入数据
path='play_tennis.csv'
df=pd.read_csv(path)
len_df=len(df)
#处理数据
#outlook: 0 rain   1 overcast   2 sunny
#tem:     0 cool   1 mild       2 hot
#hum:     0 normal 1 high
#wind    0 weak    1 strong

del df['day']

df.loc[df['outlook'] == 'Rain','outlook'] = 0
df.loc[df['outlook'] == 'Overcast','outlook'] = 1
df.loc[df['outlook'] == 'Sunny','outlook'] = 2

df.loc[df['temp'] == 'Cool','temp'] = 0
df.loc[df['temp'] == 'Mild','temp'] = 1
df.loc[df['temp'] == 'Hot','temp'] = 2

df.loc[df['humidity'] == 'High','humidity'] = 1
df.loc[df['humidity'] == 'Normal','humidity'] = 0

df.loc[df['wind'] == 'Weak','wind'] = 0
df.loc[df['wind'] == 'Strong','wind'] = 1

#转换数据类型dataframe为list
df_=np.array(df)
df_list=df_.tolist()
#计算H（D）
def calshang(df_list):
    numE=len(df_list)   #得到数目
    labelCounts={}
    for i in df_list:
        currentLable=i[-1]
        if currentLable not in labelCounts.keys():
            labelCounts[currentLable]=0
        labelCounts[currentLable]+=1

    Ent=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numE
        Ent-=prob*math.log(prob,2)
    return Ent

#按照给定特征划分数据集,便于后续经验条件熵的计算
def splitDataSet(df_list,axis,value):
    retdf=[]
    for i in df_list:
        if i[axis]==value:
            rei=i[:axis]
            rei.extend(i[axis+1:])
            retdf.append(rei)
    return retdf

#选取最优特征
def choosebest(df_list):
    H_D=calshang(df_list)   #计算经验熵
    numFeatures = len(df_list[0]) - 1  #获得行数
    bestInfoGain=0.0   #最大信息增益
    bestFeature=0      #选区的最优feature

    for i in range(0, numFeatures):
        featList = [example[i] for example in df_list]#获取每个特征下的分类
        uniqueVals = set(featList)  #合成一个列表
        newEntorpy = 0.0
        for value in uniqueVals:   #遍历，获得此特征的信息增益
            subDataSet = splitDataSet(df_list, i, value)
            prob = len(subDataSet) / float(len(df_list))
            newEntorpy += prob * calshang(subDataSet)
        infoGain = H_D - newEntorpy#信息增益
        if (infoGain > bestInfoGain):#更新，获取最佳信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#多数表决法选择不确定的叶子节点
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
                classCount[vote] +=1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

myTree={}
#建立决策树
def createtree(df_list,labels):
    classList = [example[-1] for example in df_list]
    if classList.count(classList[0]) == len(classList):  # 如果数据集样本属于同一类，说明该叶子结点划分完毕
        return classList[0]
    if len(df_list[0]) == 1:  # 如果数据集样本只有一列（该列是类标签），说明所有属性都划分完毕，则根据多数表决方法，对该叶子结点进行分类
        return majorityCnt(classList)
    bestFeat = choosebest(df_list)  # 根据信息增益，选择最优的划分方式
    bestFeatLabel = labels[bestFeat] #记录
    myTree = {bestFeatLabel: {}}  # 树
    del (labels[bestFeat])  # 在属性标签中删除该属性
    # 根据最优属性构建树
    featValues=[]
    uniqueVals=[]
    featValues = [example[bestFeat] for example in df_list]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        subDataSet = splitDataSet(df_list, bestFeat, value)
        myTree[bestFeatLabel][value] = createtree(subDataSet, subLabels)
    print("myTree:", myTree)
    return myTree

#测试
def classify(inputTree,featLabels,testVec):
    global classlable
    firstStr = list(inputTree.keys())[0]  # 树根代表的属性
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  # 树根代表的属性，所在属性标签中的位置，即第几个属性
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

n=len_df//5
global num_right
num_right=0
df_train=[]
df_test=[]

for k in range(5):
      df_list_ = []
      if k==0:
          for y in range(n, len_df):
              df_list_.append(df_list[y])
      if k!=0:
          for i in range(k*n):
            df_list_.append(df_list[i])
          for y in range((k+1)*n,len_df):
            df_list_.append(df_list[y])

      labels = ['outlook', 'temp', 'humidity', 'wind']
      labelsForCreateTree = labels[:]
      Tree = createtree(df_list_, labelsForCreateTree)
      for v in range(k * n, (k + 1) * n):
          i_ = []
          i_ = df_list[v][:4]
          testvec = i_
          if classify(Tree, labels, testvec) == df_list[v][4]:
              num_right = num_right + 1

print("正确率：", num_right / 10)








