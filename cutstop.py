import pandas as pd
import json
import os
import jieba


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, encoding='UTF-8').readlines()]
    return stopwords

def seg_sentence(sentence):
    sentence = str(sentence)
    sentence_seged = jieba.lcut(sentence.strip())
    stopwords = stopwordslist(r'C:\Users\46350\PycharmProjects\Algorithm\stopword.txt')
    outstr = []
    for word in sentence_seged:
        if word not in stopwords and word != '\t':
                outstr.append(word)
    return outstr
files = os.listdir(r'C:\Users\46350\PycharmProjects\Algorithm\simulation\sh601857')
#files = os.listdir('./precess/before/')
for file in files:
    print('正在处理------simulation/news/' + file)
    path = os.path.join(r'C:\Users\46350\PycharmProjects\Algorithm\simulation\news', file)
    path_out = os.path.join(r'C:\Users\46350\PycharmProjects\Algorithm\simulation\stop_news', file)
    #path = 'simulation/news/2022-04-05.csv'
    data = pd.read_csv(path, index_col=False, header=0)
    file_name = file.split('.')[0]
    file_name = os.path.join(r'C:\Users\46350\PycharmProjects\Algorithm\simulation\stop_news', file_name) + '.text'
    with open(file_name, 'w', encoding='utf-8') as f:
        if data.shape[1] == 2:
            news_text = seg_sentence(data.iloc[4][1])
            # data['news_text'][1]
            dict = {}
            dict['date'] = data.iloc[1][1]
            dict['news_text'] = news_text
            js = json.dumps(dict)
            f.write(js + '\n')
        else:
            for i in range(len(data)):
                news_text = seg_sentence(data['news_text'][i])
               # data['news_text'][1]
                dict = {}
                dict['date'] = data['date'][i]
                dict['news_text'] = news_text
                js = json.dumps(dict)
                f.write(js+'\n')
