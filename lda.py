# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 14:54:08 2019

@author: Administrator
"""
import re
import sys
import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation 
from gensim import corpora
from gensim import models
import jieba.posseg as pseg
from data_pre import ReadData,GetFinalData
import math

#cut_words_str,cut_words_list = cut_words(comment_all)
stopwords_path = './data/StopwordsCN.txt'
simi_path = './data/simi_words.txt'
positive_path = './data/posdict.txt'
negative_path = './data/negdict.txt'
mostdict_path = './data/mostdict.txt'
verydict_path = './data/verydict.txt'
moredict_path = './data/moredict.txt'
lshdict_path = './data/lshdict.txt'
insufficientdict_path = './data/insufficientdict.txt'

def SimiWords(simi_path):
    
    '''同义词替换制作成字典.'''
    
    #help(pd.read_csv)
    simi_ = []
    simi_words_df = pd.read_csv(simi_path,names=['classes','labels','nums'],sep='	')
    simi_words_df = simi_words_df.drop_duplicates('labels', keep='first', inplace=False)
    simi_words = simi_words_df.labels.tolist()
    simi_words_iterm = simi_words_df.groupby('classes').labels
    
    for _,syn in simi_words_iterm:
        for _ in syn.tolist():
            simi_.append(syn.tolist()[0])
    
    return dict(zip(simi_words,simi_)),simi_words


def CutWords_list(comment):
    
    '''每一条数据评论切词.后续副词正反向词权重准备.'''
    
    if type(comment) == str:
        
        words_list = list(jieba.cut(comment))
        return words_list
    else:
        return []


'''
def FeatureWords(cut_words,simi_path):
    
    #名词过滤与同义词合并得到特征词
    final_words = []
    temp = pseg.cut(cut_words)
    dict_simi,simi_words_ = SimiWords(simi_path)
    
    
    
    #num = 1 
    for i in temp:
        
        #print("正在处理第%d个词语,总计有2412421个词语。耐心等候....." % num)
        #num += 1
        if i.flag == 'n':
            if i.word in simi_words_:
                final_words.append(dict_simi.get(i.word))
            else:
                final_words.append(i.word)
    return final_words
'''


def LoadStopWord(stopwords_path):
    
    '''加载停用词'''
    
    stopwords = pd.read_csv(stopwords_path, 
        #index_col=False
    )

    return stopwords.stopword.tolist()


def GetNoun(comment_data,stopwords_path,simi_path):
    
    '''处理分词后的词语：停用词、名词过滤、相关词替换.'''
    
    count = 1
    noun_words = []
    stop_words = LoadStopWord(stopwords_path)
    dict_simi,simi_words = SimiWords(simi_path)
    #rep = dict((re.escape(k), v) for k, v in dict_simi.items())
    #pattern = re.compile("|".join(rep.keys()))
    #replace_comment = []
    for i in comment_data:        
        print("正在提取第{}条评论的名词，总计有{}条评论需要提取。耐心等候......".format(count,len(comment_data)))
        count += 1        
        if type(i) == str:
            #i = pattern.sub(lambda m: rep[re.escape(m.group(0))], i)
            #replace_comment.append(i)
            word_list = []
            for word,flag in pseg.cut(i):
                if (flag == 'n') and ( word not in stop_words) and (len(word) >= 2):
                    if word in simi_words:
                        word_list.append(dict_simi.get(word))
                    else:
                        word_list.append(word)
            noun_words.append(word_list)
        else:
            noun_words.append([])
            #replace_comment.append('')
            
    return noun_words#,replace_comment
#noun_words = GetNoun(data_all.评论,stopwords_path,simi_path)


def Perplexity(ldamodel, test_data, dictionary, size_dictionary, num_topics):
    
    '''Lda模型评估函数.'''
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word_list = []
    
    for topic_id in range(num_topics): 
        topic_word = ldamodel.show_topic(topic_id, size_dictionary) 
        dic = {} 
        for word, probability in topic_word: 
            dic[word] = probability 
            topic_word_list.append(dic) 
    doc_topics_ist = []
    for doc in test_data:
        doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
    testset_word_num = 0
    
    for i in range(len(test_data)):
        prob_doc = 0.0 # the probablity of the doc
        doc = test_data[i]
        doc_word_num = 0
        for word_id, num in doc: 
            prob_word = 0.0 
            doc_word_num += num 
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                prob_topic = doc_topics_ist[i][topic_id][1] 
                prob_topic_word = topic_word_list[topic_id][word] 
                prob_word += prob_topic*prob_topic_word
            prob_doc += math.log(prob_word)
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum/testset_word_num)
    print ("主题数为{}的Perplexity为: {}。".format(num_topics,int(prep)))

    return prep


def GetBestLdaModel(n_words):
    
    '''lda模型获得特征值.'''
    best_prep = 1000000
    best_lda = None
    best_num_topic = None
    num_topics = [i for i in range(5,30)]
    words_test = n_words[:int(len(n_words)/3)]
    dic = corpora.Dictionary(n_words)
    corpus = [dic.doc2bow(text) for text in n_words]
    test_corpus = [dic.doc2bow(text) for text in words_test]
    for num_topic in num_topics:
        lda = models.LdaModel(corpus,id2word=dic,num_topics=num_topic)
        prep = Perplexity(lda, test_corpus, dic,len(dic.keys()), num_topic)
        if prep <= best_prep:
            best_prep = prep
            best_lda = lda
            best_num_topic = num_topic
    print("最好的主题数是：%d，其Perplexity是：%d" %(best_num_topic,best_prep))
    ldaout = best_lda.print_topics(num_topics=best_num_topic)
    
    return ldaout
#ldaout = GetFeatureWord_All(noun_words)


def ChooseFeatureWord(ldaout,s):
    
    '''阈值s，提取特征值.'''
    
    feature_words = []
    for i in ldaout:
        temp = i[1].replace('"','').replace(' ','').split("+")
        for temp_ in temp:
            if eval(temp_.split("*")[0]) >= s:
                print(temp_.split("*")[1])
                feature_words.append(temp_.split("*")[1])

    return feature_words
#feature_words = ChooseFeatureWord(ldaout,0.05)
#test = data_all.评论.apply(CutWords_list)


def GetDict(data,weight):
    
    '''权重字典.'''
    
    return dict(zip(data,[weight for i in data]))


def GetFeatureWordData(positive_path,negative_path,mostdict_path,verydict_path,moredict_path,lshdict_path,insufficientdict_path):
    
    '''处理特征词权值等数据返回list,dict.'''
    
    poslist = ReadData(positive_path).posdict.tolist()
    neglist = ReadData(negative_path).negdict.tolist()
    mostlist = ReadData(mostdict_path).mostdict.tolist()
    verylist = ReadData(verydict_path).verydict.tolist()
    morelist = ReadData(moredict_path).moredict.tolist()
    lshlist = ReadData(lshdict_path).lshdict.tolist()
    insufficientlist = ReadData(insufficientdict_path).insufficientdict.tolist()
    
    e_list = poslist + neglist
    e_dict = dict(GetDict(poslist,1),**GetDict(neglist,-1))
    
    adv_list = mostlist + verylist + morelist + lshlist + insufficientlist
    adv_dict = dict(dict(dict(dict(GetDict(mostlist,2),**GetDict(verylist,1.5)),**GetDict(morelist,1.25)),**GetDict(lshlist,0.5)),**GetDict(insufficientlist,0.25))
    
    return e_list,e_dict,adv_list,adv_dict
#comments_list = test.tolist()


def GetCommentScore(feature_words,comments_list):
    
    '''得到所有评论的各个特征词的得分情况。返回值 df.'''
    
    comment_score = pd.DataFrame(data=np.ones((len(comments_list),len(feature_words))),columns=feature_words)
    e_list,e_dict,adv_list,adv_dict = GetFeatureWordData(positive_path,negative_path,mostdict_path,verydict_path,moredict_path,lshdict_path,insufficientdict_path)
    for i in range(len(comments_list)):
        print("正在处理第%d条评论的得分情况，总计有%d条评论需要处理。" %(i,len(comments_list)))
        for feature in feature_words:
            feature_weight = 1
            feature_score = 0
            if feature in comments_list[i]:
                curr_index = comments_list[i].index(feature)
                try:
                    if comments_list[i][curr_index + 1] in e_list:
                        curr_score = e_dict.get(comments_list[i][curr_index + 1])
                        feature_score = feature_score + curr_score
                except IndexError:
                    print("向后无情感词")
                    
                try:
                    if comments_list[i][curr_index + 2] in adv_list:
                        curr_weight = adv_dict.get(comments_list[i][curr_index + 2])
                        feature_weight = feature_weight * curr_weight
                except IndexError:
                    print("向后无副词")
                    
                    
            comment_score.loc[i,feature] = feature_weight * feature_score
    comment_score.to_csv(r"C:\Users\Administrator\Desktop\DM\comment_score_1.csv",index=False,encoding='utf_8_sig')    
    
    return comment_score       
#comment_score = GetCommentWeight(feature_words,comments_list)


def GetModelData():
    
    '''合并原始数据和特征词得分数据,得到模型要训练的数据.'''
    
    data_all,name_dict = GetFinalData()
    noun_words = GetNoun(data_all.评论,stopwords_path,simi_path)
    ldaout = GetBestLdaModel(noun_words)
    feature_words = ChooseFeatureWord(ldaout,0.05)
    comments_list = data_all.评论.apply(CutWords_list).tolist()
    comment_score = GetCommentScore(feature_words,comments_list)
    data_all.drop('评论',axis=1,inplace=True)
    data_all = data_all.join(comment_score)
    #data_all.index = [i for i in range(len(data_all))]

    return data_all