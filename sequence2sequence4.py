#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2018/4/25 10:40
@Author  : Junya Lu
@Site    : 
"""

from Word2Vec import Word2Vec
import jieba
from gensim import corpora ,models,similarities
import jieba
from collections import defaultdict

# 读文件
fr = open('train.txt', 'r')
fr2 = open('test.txt', 'r')

#1.导入句子

file_num = -1 # 分批处理
lines = fr.readlines()
test_lines = fr2.readlines()
print len(lines)
while(file_num<490000):
    texts = []
    i = 0
    dic_source = {}
    filename = 'submit426_new2.txt'
    # filename = 'submit' + str(file_num) + '.txt'
    fw = open(filename, 'w')
    print lines[file_num+1]
    # 导入句子
    for line in lines:
        line_str = line.strip().split('\t')
        tid = line_str[0]
        sequence = ' '.join(line_str[1:])
        dic_source[i] = int(tid)
        i = i + 1
        line_str = sequence
        data = "".join(jieba.cut(line_str))
        texts.append(list(data))

    # 4.基于文本建立字典
    dictionary = corpora.Dictionary(texts)
    featureNum = len(dictionary.token2id.keys())  # 提取词典特征数
    dictionary.save("./dict_LearnGensim.txt")

    # 5.基于词典建立新的语料库
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 6.TF-IDF 处理
    tfidf = models.TfidfModel(corpus)
    # index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_best=21, num_features=featureNum) # 0.06864
    index = similarities.Similarity(tfidf[corpus], num_best=21, num_features=featureNum)
    index.save('./Similarity.index')
    # index = similarities.MatrixSimilarity.load('./MatrixSimilarity.index')
    print 'starting written'
    for line in test_lines:
        line_str = line.strip().split('\t')
        target_id = line_str[0]
        query = ' '.join(line_str[1:])
        # 7.加载句子并整理其格式
        # target_id = 1
        # query = "美国地质勘探局：印度莫黑安东南部151公里处发生5.1级地震"

        dataQ = "".join(jieba.cut(query))
        dataQuery = ""
        for item in dataQ:
            dataQuery += item + " "
        new_doc = dataQuery

        # 8.将对比句子转换为稀疏向量
        new_vec = dictionary.doc2bow(new_doc.split())

        # 9.计算相似性


        sim = index[tfidf[new_vec]]
        sorted_sim = sorted(sim, reverse=True)
        print sim
        # print sorted_sim
        for i in sim:
            if int(target_id) != int(dic_source[i[0]]):
                fw.write(str(target_id) + '\t' + str(dic_source[i[0]]) + '\t' + str(i[1]) + '\n')
            else:
                continue

        # for i in range(20):
        #     print("查询与第%d句话相似度为%f" % (sim.index(sorted_sim[i]) + 1, sorted_sim[i]))
        #     print("查询与id %d句话相似度为%f" % (dic_source[sim.index(sorted_sim[i]) + 1], sorted_sim[i]))
        #     fw.write(str(target_id) + '\t' + str(dic_source[sim.index(sorted_sim[i]) + 1]) + '\t' + str(
        #         sorted_sim[i]) + '\n')

    fw.close()
    print file_num,  filename, 'has been written!!!'
    file_num += 490005

