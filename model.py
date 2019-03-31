import sys
import pandas as pd
import numpy as np
import datetime
from lda import GetModelData
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import xgboost


def FillMean(data_s):
    '''输入series数据，返回填充均值的series'''

    return data_s.fillna(data_s.mean())


def FillNa(data):
    '''
       预测排名2.
       删除排名2为空的数据.
       价格1、星级、评论数量 使用均值填充.
       价格2 填充对应的价格1.
    '''

    data = data[data.排名2.notna()].reset_index(drop=True)
    data.价格1 = FillMean(data.价格1)
    data.星级 = FillMean(data.星级)
    data.评论数量 = FillMean(data.评论数量)
    indexes = np.where(np.isnan(data.价格2))[0]
    data.loc[indexes, '价格2'] = data.loc[indexes, '价格1']

    return data


def GetColumns(data, mode):
    '''
    mode 参数：
        None: 返回源数据.
        PCC: 皮尔逊相关系数，返回相关系数大于0.05的字段.
        PCA: pca降维字段。默认返回18个字段.
    '''

    if mode == None:
        y = data.排名2
        data.drop(['排名2'], axis=1, inplace=True)

    elif mode == 'PCC':
        columns_corr_ = np.abs(data.corr().排名2)
        data = data[list(columns_corr_[columns_corr_ >= 0.005].index)]
        y = data.排名2
        data.drop(['排名2'], axis=1, inplace=True)

    elif mode == 'PCA':
        y = data.排名2
        data.drop(['排名2'], axis=1, inplace=True)
        data = pd.DataFrame(data=PCA(n_components=18).fit_transform(
            data), columns=['PCA_' + str(i) for i in range(18)])

    return data, y


def PreData(data, mode=None, group=False):
    '''
    数据预处理：采用什么模式处理字段、是否对字段名称分组、归一化数据、切分数据集(0.3).
    '''

    x = FillNa(data)
    if group == True:
        x = x.groupby('名称').mean().reset_index(drop=True)
    else:
        x.drop(['名称'], axis=1, inplace=True)
    x, y = GetColumns(x, mode)
    columns = x.columns
    x = pd.DataFrame(data=StandardScaler().fit_transform(x), columns=columns)

    return train_test_split(x, y, test_size=0.3, random_state=0)
