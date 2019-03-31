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
