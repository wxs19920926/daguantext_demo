# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 10:49:19 2018

@author: wxs
"""
from os import path
from .. import config
import logging
import tensorflow as tf
import numpy as np
logger = logging.getLogger('train.dataset')
from collections import Counter

current_path = path.dirname(__file__)
parent_path  = path.dirname(current_path)

#读取文件
def load_files(files = None):
    """
    加载文件
    """
    logger.info("loading data from many text files ...")
    
    texts_trains = list()
    texts_tests = list()
    
    for fid, file in enumerate(files):
        f = open(file, encoding="utf-8")
#        file_datas = f.readlines()
        if fid == 0:
            file_datas = f.readlines()
            for file_data in file_datas[1:5]:
                texts_train = list()
                text = file_data.split(',')
#                texts_train.append(text[0].strip())
                texts_train.append(text[1].split(' '))
                texts_train.append(text[2].split(' '))
                texts_train.append(text[3].strip())
                texts_trains.append(texts_train)
#        else:
#            for file_data in file_datas[1:]:
#                texts_test = dict()
#                text = file_data.split(',')
#                texts_test['id'] = text[0]
#                texts_test['article'] = text[1].split(' ')
#                texts_test['word_seg'] = text[2].split(' ')
#                texts_test['wclass'] = text[3]
#                texts_tests.append(texts_test)
    return texts_trains, texts_tests

#数据集整理
def build_dataset(texts_dataset : list, index : int):
    texts_dataset = np.array(texts_dataset)
    texts = texts_dataset[:, index]
    labels = texts_dataset[:, 2]
    return texts, labels

#去除空的数据集
def drop_empty_texts(texts, labels):
    """
    去除预处理后句子为空的评论
    :param texts: id形式的文本列表
    :return: tuple of arrays. 非空句子列表，非空标记列表
    """
    logger.info("clear empty sentences ...")
    non_zero_idx = [id_ for id_, text in enumerate(texts) if len(text) != 0]
    texts_non_zero = np.array([texts[id_] for id_ in non_zero_idx])
    labels_non_zero = np.array([labels[id_] for id_ in non_zero_idx])
    return texts_non_zero, labels_non_zero

def dataset_padding(text_ids, sent_len):
    """
    句子id列表左侧补0
    :param text_ids: id形式的句子列表
    :param seq_ken:  int, 最大句长
    :return: numpy array.  补0后的句子
    """
    logger.info("padding dataset ...")
    textids_padded = np.zeros((len(text_ids), sent_len), dtype=int)
    for i, row in enumerate(text_ids):
        textids_padded[i, -len(row):] = np.array(row)[:sent_len]

    return np.array(textids_padded)

def dataset_split(texts, labels, random_seed=None):
    """
    训练、开发、测试集划分，其中训练集比例为train_percent，开发集和测试各集为0.5(1-train_percent)
    :param text: 数据集x
    :param labels: 数据集标记
    :return: (val_x, val_y, test_x, test_y)
    """
    logger.info("split dataset ...")
    # 检测x与y长度是否相等
    assert len(texts) == len(labels)
    # 随机化数据
    if random_seed:
        np.random.seed(random_seed)
    shuf_idx = np.random.permutation(len(texts))
    texts_shuf = np.array(texts)[shuf_idx]
    labels_shuf = np.array(labels)[shuf_idx]

#    # 切分数据
#    split_idx = int(len(texts_shuf)*train_percent)
#    train_x, val_x = texts_shuf[:split_idx], texts_shuf[split_idx:]
#    train_y, val_y = labels_shuf[:split_idx], labels_shuf[split_idx:]

    test_idx = int(len(texts_shuf)*0.5)
    val_x, test_x = texts_shuf[:test_idx], texts_shuf[test_idx:]
    val_y, test_y = labels_shuf[:test_idx], labels_shuf[test_idx:]

    return val_x, val_y, test_x, test_y

#统计类别
def countclass_num(labels : list):
    class_counts = Counter(labels)
    class_num = sorted(class_counts, key=class_counts.get, reverse=True)
    return len(class_num)

def labels2onehot(labels, class_num):
    """
    生成句子的情感标记
    :param labels: list of labels. 标记列表
    :param class_num: 类别总数
    :return: numpy array.
    """
    def label2onehot(label_, class_num):
        onehot_label = [0] * class_num
        onehot_label[label_] = 1
        return onehot_label
    return np.array([label2onehot(label_, class_num) for label_ in labels])
    

def main():
    texts_trains, texts_tests = load_files([parent_path + '/data/origin/train_set.csv', parent_path + '/data/origin/test_set.csv'])
    texts, labels = build_dataset(texts_trains, 1)
    texts_test, labels_test = build_dataset(texts_tests, 1)
    texts, labels = drop_empty_texts(texts, labels)
    texts_test, labels_test = drop_empty_texts(texts_test, labels_test)
    texts = dataset_padding(texts, config.max_sent_len)
    texts_test = dataset_padding(texts_test, config.max_sent_len)
    class_num = countclass_num(labels)
    labels = labels2onehot(labels,class_num)
    train_x = texts
    train_y = labels
    val_x, val_y, test_x, test_y = dataset_split(texts_test, labels_test)
    return train_x, train_y, val_x, val_y, test_x, test_y
    
    
if __name__ == '__main__':
    main()