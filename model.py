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
