# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 09:51:11 2019

@author: wxs
"""
from sklearn import neighbors,datasets,preprocessing
import sklearn.decomposition as sk_decomposition
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=0.3)
#
#pca = sk_decomposition.PCA(n_components='mle',whiten=False,svd_solver='auto')
#iris_X = iris.data
#pca.fit(iris_X)
#reduced_X = pca.transform(iris_X) #reduced_X为降维后的数据
#print('PCA:')
#print ('降维后的各主成分的方差值占总方差值的比例',pca.explained_variance_ratio_)
#print ('降维后的各主成分的方差值',pca.explained_variance_)
#print ('降维后的特征数',pca.n_components_)
#print ('降维后的特征数',reduced_X)

#import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from sklearn.datasets.samples_generator import make_blobs
## X为样本特征，Y为样本簇类别， 共1000个样本，每个样本3个特征，共4个簇
#X, y = make_blobs(n_samples=10000, n_features=3, centers=[[3,3, 3], [0,0,0], [1,1,1], [2,2,2]], cluster_std=[0.9, 0.1, 0.2, 0.2], 
#                  random_state =9)
#fig = plt.figure()
#ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
#plt.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o')

#import sklearn.linear_model as sk_linear
#model = sk_linear.LinearRegression(fit_intercept=True,normalize=False,copy_X=True,n_jobs=1)
#model.fit(X_train,y_train)
#acc=model.score(X_test,y_test) #返回预测的确定系数R2
#print('线性回归:')
#print('截距:',model.intercept_) #输出截距
#print('系数:',model.coef_) #输出系数
#print('线性回归模型评价:',acc)

import sklearn.linear_model as sk_linear
model = sk_linear.LogisticRegression(penalty='l2',dual=False,C=1.0,n_jobs=1,random_state=20,fit_intercept=True)
model.fit(X_train,y_train) #对模型进行训练
acc=model.score(X_test,y_test) #根据给定数据与标签返回正确率的均值
print('逻辑回归模型评价:',acc)

