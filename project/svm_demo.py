# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 15:52:37 2019

@author: wxs
"""

"""
@简介：tfidf特征/ SVM模型
@成绩： 0.77
"""
#导入所需要的软件包
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
#import sys
#import csv
#csv.field_size_limit(sys.maxsize)

print("开始...............")

#====================================================================================================================
# @代码功能简介：从硬盘上读取已下载好的数据，并进行简单处理
# @知识点定位：数据预处理
#====================================================================================================================


#df = pd.read_csv(open('./data/origin/train_set.csv', 'rU'), engine='python')  # 数据读取
df = pd.read_csv('./data/origin/train_set.csv', nrows = 4000)  # 数据读取

#df_test = pd.read_csv('./data/origin/test_set.csv', nrows = 1000)
df_train = df[:len(df) - 1000]
df_test = df[len(df) - 1000:]

x_train, x_test, y_train, y_test = train_test_split(df[['word_seg']], df[['class']], test_size=0.1, random_state=1)

# 观察数据，原始数据包含id、article(原文)列、word_seg(分词列)、class(类别标签)
#df_train = X_train.drop(['article', 'id'], axis=1) # drop删除列
#df_test = X_test.drop(['article'], axis=1)

#==========================================================
# @代码功能简介：将数据集中的字符文本转换成数字向量，以便计算机能够进行处理（一段文字 ---> 一个向量）
# @知识点定位：特征工程
#==========================================================
#vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1) 
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)
'''
    ngram_range=(1, 2) : 词组长度为1和2
    min_df : 忽略出现频率小于3的词
    max_df : 忽略在百分之九十以上的文本中出现过的词
'''
vectorizer.fit(df['word_seg'])  # 构造tfidf矩阵
x_train = vectorizer.transform(x_train['word_seg'])  # 构造训练集的tfidf矩阵
x_test = vectorizer.transform(x_test['word_seg'])  # 构造测试的tfidf矩阵

y_train = y_train['class']-1 #训练集的类别标签（减1方便计算）

#==========================================================
# @代码功能简介：训练一个分类器
# @知识点定位：传统监督学习算法之线性逻辑回归模型
#==========================================================

classifier = LinearSVC()  # 实例化逻辑回归模型
classifier.fit(x_train, y_train)  # 模型训练，传入训练集及其标签

#根据上面训练好的分类器对测试集的每个样本进行预测
y_test_pred = classifier.predict(x_test)

#评估分数
accuracy = accuracy_score(y_true=y_test-1, y_pred=y_test_pred)
print(accuracy)


fp = open('./data/origin/data_tfidf.pkl', 'wb')
pickle.dump(classifier, fp)

#f3 = open('./data_tfidf.pkl', 'rb')
#classifier = pickle.load(f3)
#f3.close()

#将测试集的预测结果保存至本地
#df_test['class_r'] = y_test.tolist()
#df_test['class_r'] = df_test['class_r'] + 1
#df_result = df_test.loc[:, ['id', 'class', 'class_r']]
#df_result.to_csv('./data/origin/beginner.csv', index=False)

print("完成...............")