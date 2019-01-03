# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 14:21:15 2019

@author: wxs
"""
from .train import dataset
from .model import dnn_model
from . import config

# ================== step0: 定义超参数 =================
learning_rate = 0.05     # 学习率
batch_size = 256        # mini-batch大小
refine = False          # 词向量矩阵是否参与训练
epochs = 10              # 数据迭代次数
lstm_sizes = [128, 64]  # 各层lstm的维度
embed_size = 300        # 词向量维度
keep_prob = 0.8         # drop out 保留率
max_sent_len = 60      # 最大句长
l2reg = 0.000            # l2正则化参数

maxword = 5000            # 最大单词量
class_num = 3           # 类别数量
lang = 'EN'             # 文本语言 EN为英文，CN为中文
train_percent = 0.8     # 训练数据的比例
show_step = 2          # 每隔几个批次输出一次结果
embedding_matrix = None  # 设词向量矩阵为None

if __name__ == '__main__':
    train_x, train_y, val_x, val_y, test_x, test_y = dataset.main()
    model = dnn_model.DNNModel(class_num=class_num, batch_size=batch_size,
                    embed_dim=embed_size, rnn_dims=lstm_sizes,
                    vocab_size=config.max_sent_len, embed_matrix=embedding_matrix,
                    l2reg=l2reg, refine=refine)
    
    model.build()
    model.train(train_x, train_y, val_x, val_y, learning_rate, epochs, keep_prob, show_step=show_step)
    