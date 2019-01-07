# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 16:14:08 2019

@author: wxs
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score


#corpus = [
#    'This is the first document.',
#    'This is the second second document.',
#    'He And the third one.',
#    'Is this the first document?',
#]
#
#vectorizer = CountVectorizer()
#count = vectorizer.fit_transform(corpus)
#print(vectorizer.get_feature_names())  
#print(vectorizer.vocabulary_)
#print(count.toarray())
#
#transformer = TfidfTransformer()
#tfidf_matrix = transformer.fit_transform(count)
#print(tfidf_matrix.toarray())

#=====================================================

#import jieba
#
#text = """我是狗！
#    我把月，
#    我把日，
#    我把星球，
#    我把宇宙。
#    我是我！"""
#sentences = text.split()
#sent_words = [list(jieba.cut(sent0)) for sent0 in sentences]
#document = [" ".join(sent0) for sent0 in sent_words]
#
#vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
#count = vectorizer.fit_transform(document)
#print(vectorizer.get_feature_names()) 
#print(count.toarray())
#
#tfidf_model3 = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=1).fit(document)  
#print(tfidf_model3.vocabulary_)
#sparse_result = tfidf_model3.transform(document)     # 得到tf-idf矩阵，稀疏矩阵表示法
#print(sparse_result.todense())

#=========================================================
# y_pred是预测标签
y_pred, y_true=[1,5,3,4], [2,2,3,4]
accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
print(accuracy)
