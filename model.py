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

def LRModel(x_train,x_test,y_train,y_test):
    
    '''训练LR模型.'''
    
    model = LinearRegression()
    start_time = datetime.datetime.now()
    model.fit(x_train,y_train)
    end_time = datetime.datetime.now()
    print("线性回归各参数：")
    print(model.coef_)
    print("\n")
    y_pred = model.predict(x_test)
    r2_loss = r2_score(y_test,y_pred)
    print("线性回归的r方值为{}".format(r2_loss))
    print("LR模型总计用时: %d s" % (end_time- start_time).seconds)
    
    return model,r2_loss


def SVMModel(x_train,x_test,y_train,y_test):
    
    '''SVM模型，采用网格搜索超参数.'''
    
    model = SVR()
    start_time = datetime.datetime.now()
    #help(SVR)
    param_grid = {'C': [0.01,0.1,1,10,100],
                  'gamma': [0.1, 1, 10],
                  'kernel': ['linear','rbf'],
                  }
    #help(GridSearchCV)
    grid_model= GridSearchCV(model, param_grid, cv=5,scoring='r2',n_jobs=-1)
    grid_model.fit(x_train,y_train)
    end_time = datetime.datetime.now()
    print("SVM最优参数如下：")
    print(grid_model.best_params_)
    y_pred = grid_model.predict(x_test)
    r2_loss = r2_score(y_test,y_pred)
    print("SVM在验证集上的r方值为{}".format(r2_loss))
    print("SVM网格搜索超参数总计用时: %d s" % (end_time- start_time).seconds)
    
    return grid_model,r2_loss


def XBModel(x_train,x_test,y_train,y_test):
    
    '''Xgboost模型，采用网格搜索超参数.'''
    
    model = xgboost.XGBRegressor(n_jobs=-1)
    start_time = datetime.datetime.now()
    param_grid = {'learning_rate': [0.5,0.1,0.05,0.01], 
                      'n_estimators': [400,600,800,1000,1200,1600], 
                      'max_depth': [3,4,5], 
                      }
    grid_model= GridSearchCV(model, param_grid, cv=5,scoring='r2',n_jobs=-1)
    grid_model.fit(x_train,y_train)
    end_time = datetime.datetime.now()
    print("xgboost最优超参数如下：")
    print(grid_model.best_params_)
    y_pred = grid_model.predict(x_test)
    r2_loss = r2_score(y_test,y_pred)
    print("xgboost在验证集上的r方值为{}".format(r2_loss))
    print("Xgboost网格搜索超参数总计用时: %d s" % (end_time- start_time).seconds)
    
    return grid_model,r2_loss

def PlotResult(model,x_test,y_test):
    
    '''预测值\真实值可视化.'''
    
    x = [i+1 for i in range(len(y_test))][:100]
    y_pred = model.predict(x_test)
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig = plt.figure(figsize=(12,6))
    plt.plot(x,y_test[:100],c='green',label='y_true')
    plt.plot(x,y_pred[:100],c='red',label='y_pred')
    plt.legend()
    plt.xlabel("预测\真实")
    plt.ylabel("值大小")
    plt.title("预测值与真实值变化趋势图")
    plt.show()


if __name__ == '__main__':
    
    '''
    三种基本模型：LRmodel、Xgboost、SVM
    三种基本模式：PCA、PCC、None
    是否对数据分组：True、False
    通过以上对数据进行搭配训练。
    '''
    model_data = GetModelData()
    x_train,x_test,y_train,y_test = PreData(model_data,mode='PCA',group=False)
    lr_model,lr_r2_loss = LRModel(x_train,x_test,y_train,y_test)
    PlotResult(lr_model,x_test,y_test) 
    #svm_model,svm_r2_loss,x_pred = SVMModel_(x_train,x_test,y_train,y_test)
    #PlotResult(svm_model,x_test,y_test)   
    #xgboost_model,xgboos_r2_loss = XBModel(x_train,x_test,y_train,y_test)
    #PlotResult(xgboost_model,x_test,y_test)