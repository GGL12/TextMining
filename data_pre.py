# -*- coding: utf-8 -*-
"""
Created on Sat Feb  10 21:48:06 2019

@author: Administrator
"""
import jieba
import pandas as pd
import numpy as np
import re
import jieba.posseg as pseg

#text_test = '这款手机质量好，就是价钱有点贵。'
# len(list(pseg.cut(cut_words_str)))
stopwords_path = './data/StopwordsCN.txt'
data_1_path = './data/data1.csv'
data_2_path = './data/data2.csv'
data_3_path = './data/data3.csv'
data_4_path = './data/data4.csv'


def ReadData(data_path):
    '''读取数据.'''

    return pd.read_csv(data_path, index_col=False)


def ConcatData(path1, path2, path3, path4):
    '''合并数据.'''

    data_1 = ReadData(path1)
    data_2 = ReadData(path2)
    data_3 = ReadData(path3)
    data_4 = ReadData(path4)
    data_all = pd.concat([data_1, data_2, data_3, data_4]
                         ).reset_index(drop=True)

    return data_all


def GetNum(x):
    '''
    适用于价格1，星级，评论数量 字段提取对应的数字信息.
    '''

    if type(x) == str:
        x = x.replace(',', '')
        return eval(re.findall('\d+\.?\d*', x)[0])
    else:
        return np.nan


def GetPrice_2(x):
    '''
    适用于字段价格2字段。特殊，包含了一些其他无关不出来的东西.
    如：此商品仅剩 1 件 - 欲购从速.
    '''

    if type(x) == str:

        if "￥" in x:
            x = x.replace(',', '')
            return eval(re.findall('\d+\.?\d*', x)[0])
        else:
            return np.nan
    else:
        return np.nan


def NameId(x):
    '''虚拟变量——替换商品名——[i for i in range(x)].'''

    temp = list(x.名称.unique())
    temp_dict = dict(zip(temp, [i for i in range(len(temp))]))
    x.名称 = x.名称.map(temp_dict)

    return x, temp_dict


def CutWords(words):
    '''切词并去掉停用词.'''

    cut_words = list(jieba.cut(words.replace(' ', '')))

    cut_words_df = pd.DataFrame(
        {'words': cut_words}
    )

    # 移除停用词
    stopwords = pd.read_csv(
        stopwords_path,
        # index_col=False
    )

    new_words = cut_words_df[
        ~cut_words_df.words.isin(stopwords.stopword)
    ]

    cut_words_list = new_words.words.values.tolist()

    return ''.join(cut_words_list)


'''
def DataGroupBy(x): 
    
    ''''''
    
    final_data = pd.DataFrame(columns=['名称','评论'])
    i = 0
    
    for x,y in x.groupby('名称'):
        
        class_comments = ''
        for comment in y.评论:
            if type(comment) != str:
                
                class_comments += ''
            else:
                class_comments += comment

        final_data.loc[i,'名称'] = x
        final_data.loc[i,'评论']= CutWords(class_comments)
        i += 1
        
    return final_data
'''


def DataGroupBy(x):
    '''返回每条数据分组后的均值.'''

    return x.groupby("名称").mean()


def GetAllComment(data):
    '''得到所有评论数据.'''

    comments = ''
    for comment in data.评论:
        if type(comment) == str:
            comments += comment
        else:
            comments += ''

    return comments


def GetFinalData():
    '''所有数据经过数据清洗得到最终的数据.'''

    data_all = ConcatData(data_1_path, data_2_path, data_3_path, data_4_path)
    data_all, name_dict = NameId(data_all)
    data_all.价格2 = data_all.价格2.apply(GetPrice_2)
    columns = ['价格1', '星级', '评论数量', '排名1']
    for i in columns:
        data_all[i] = data_all[i].apply(GetNum)

    return data_all, name_dict


if __name__ == '__main__':

    data_all, name_dict = GetFinalData()
    comment_df = DataGroupBy(data_all)
    comment_all = GetAllComment(data_all)
