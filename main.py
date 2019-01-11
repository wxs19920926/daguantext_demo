# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 14:21:15 2019

@author: wxs
"""
from project.train import dataset
from project.model import dnn_model
from project import config
import pickle
#import argparse

# ================== step0: 定义超参数 =================
learning_rate = 0.05     # 学习率
batch_size = 2        # mini-batch大小
refine = False          # 词向量矩阵是否参与训练
epochs = 10              # 数据迭代次数
lstm_sizes = [128, 64]  # 各层lstm的维度
embed_size = 300        # 词向量维度
keep_prob = 0.8         # drop out 保留率
#max_sent_len = 60      # 最大句长
l2reg = 0.000            # l2正则化参数

#maxword = 5000            # 最大单词量
#class_num = 3           # 类别数量
lang = 'EN'             # 文本语言 EN为英文，CN为中文
train_percent = 0.8     # 训练数据的比例
show_step = 2          # 每隔几个批次输出一次结果
embedding_matrix = None  # 设词向量矩阵为None
train_percent = 0.8     # 训练数据的比例

if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument("-m", "--mode", type=int, help="the base")
#    args = parser.parse_args()
#    train_x, train_y, val_x, val_y, test_x, test_y, class_num = dataset.main(args.mode, train_percent)
    f_emb = open('./project/word2vec/word_seg_vectors_arr.pkl', 'rb')
    emb_arr = pickle.load(f_emb)
    f_emb.close()
    f_train = open('./project/data/pro/df_train.pkl', 'rb')
    f_vali = open('./project/data/pro/df_vali.pkl', 'rb')
    df_train = pickle.load(f_train)
    df_vali = pickle.load(f_vali)
    f_train.close()
    f_vali.close()
    """shuffle数据，并to_list"""
    df_train = df_train.sample(frac=1)
    train_x = list(df_train.loc[:, 'word_seg'])
    train_y = list(df_train.loc[:, 'class'])
    
    """shuffle数据，并to_list"""
    df_train = df_vali.sample(frac=1)
    val_x = list(df_vali.loc[:, 'word_seg'])
    val_y = list(df_vali.loc[:, 'class'])
    
    train_x = dataset.dataset_padding(train_x, 2000)
    val_x = dataset.dataset_padding(val_x, 2000)
    
#    train_x, train_y, val_x, val_y, test_x, test_y, class_num = dataset.main(1, train_percent)
    model = dnn_model.DNNModel(class_num=20, batch_size=batch_size,
                    embed_dim=embed_size, rnn_dims=lstm_sizes,
                    vocab_size=config.max_words, embed_matrix=emb_arr,
                    l2reg=l2reg, refine=refine)
    
    model.build()
    model.train(train_x, train_y, val_x, val_y, learning_rate, epochs, keep_prob, show_step=show_step)
    