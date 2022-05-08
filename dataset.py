import json
import os
from datetime import date, datetime, timedelta
import numpy as np
from torch.utils import data

from torch.nn.parameter import Parameter


class Dataset(object):
    def __init__(self):
        #self.flags = flags
        self.PAD = '<PAD>'
        self.UNK = '<UNK>'
        self.word2idx = dict()
        self.idx2word = dict()
        self.word2idx[self.PAD] = 0
        self.idx2word[0] = self.PAD
        self.word2idx[self.UNK] = 1
        self.idx2word[1] = self.UNK
        self.date_min, self.date_max, self.max_date_len, self.max_news_len, self.word2vec = self.build_vocab()
        self.num_UNK_words = 0
        self.num_words = 0
        self.empty_news = None
        self.empty_news = [self.word2idx[self.PAD]] * self.max_news_len

        self.idx2label = {1: 'UP', 0: 'DOWN'}
        self.label2idx = {'UP': 1, 'DOWN': 0}

        self.stock_dict = self.load_stock_history()
        self.date_tweets = self.load_tweet()

        self.train_x, self.train_y, self.dev_x, self.dev_y, self.test_x, self.test_y = self.map_stocks_tweets()
        self.class_weights = get_class_weights(len(self.train_x), self.train_y)

    def build_vocab(self):
        '''
        :param input_dir:
        :return: date_min, date_max, max_date_len, max_news_len, np.asarray(word2vec)
        '''
        date_min = date(9999, 1, 1)
        date_max = date(1, 1, 1)
        datetime_format = '%Y-%m-%d'
        date_freq_dict = {}
        max_news_len = 0
        word_freq_dict = {}
        #???
        news_files = os.listdir(r'C:\Users\46350\PycharmProjects\Algorithm\simulation\allnews')
        for news_file in news_files:
            stock_files = os.path.join(r'C:\Users\46350\PycharmProjects\Algorithm\simulation\allnews', news_file)
            files = os.listdir(stock_files)
            for file in files:
                file = os.path.join(stock_files, file)
                with open(file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line_dict = json.loads(line)
                        text = line_dict['news_text']
                        #统计词频
                        for w in text:
                            if w in word_freq_dict:
                                word_freq_dict[w] += 1
                            else:
                                word_freq_dict[w] = 1
                        #最大新闻长度
                        text_len = len(text)
                        if max_news_len < text_len:
                            max_news_len = text_len
                        #新闻发生的最早最晚时间
                        de = line_dict['date']
                        de = datetime.strptime(de, datetime_format)
                        de = de.date()
                        if date_max < de:
                            date_max = de
                        elif date_min > de:
                            date_min = de
                        #一定时间内新闻数
                        stock_date_key = '{}'.format(de)
                        if stock_date_key in date_freq_dict:
                            date_freq_dict[stock_date_key] += 1
                        else:
                            date_freq_dict[stock_date_key] = 1
        #读取预训练好的词向量
        word2vec_dict = dict()
        word_embed_path = r'C:\Users\46350\PycharmProjects\Algorithm\simulation\sgns.financial.bigram'
        with open(word_embed_path, 'r', encoding='utf-8') as f:
            for line in f:
                cols = line.split()
                if cols[0] in word_freq_dict:
                    word2vec_dict[cols[0]] = [float(l) for l in cols[1:]]
        #最高频词
        most_freq_words = sorted(word_freq_dict, key=word_freq_dict.get,
                                 reverse=True)
        #创建word2vec
        for w in most_freq_words:

            if w not in word2vec_dict:
                continue

            w_idx = len(self.word2idx)
            self.word2idx[w] = w_idx
            self.idx2word[w_idx] = w
        final_size = len(self.word2idx)

        word2vec = list()
        sample_vec = word2vec_dict['公司'] #怎么选择？？
        word2vec.append([0.] * len(sample_vec))  # <PAD>
        word2vec.append([1.] * len(sample_vec))  # <UNK>
        for w_idx in range(2, final_size):
            word2vec.append(word2vec_dict[self.idx2word[w_idx]])
        #高频新闻日期及长度
        most_freq_news_date = sorted(date_freq_dict, key=date_freq_dict.get, reverse=True)[0]
        max_date_len = date_freq_dict[most_freq_news_date]

        return date_min, date_max, max_date_len, max_news_len, np.asarray(word2vec)

    def load_tweet(self):
        '''

        :return: date_tweets eg:{'600028 2022-01-10':[word_idx]}
        '''
        date_tweets = dict()
        num_tweets = 0
        news_files = os.listdir(r'C:\Users\46350\PycharmProjects\Algorithm\simulation\allnews')
        for news_file in news_files:
            stock_files = os.path.join(r'C:\Users\46350\PycharmProjects\Algorithm\simulation\allnews', news_file)
            for root, subdirs, files in os.walk(stock_files):
                # stock_name  = str(root).replace('E:\vscode\project\HAN\precess\\', '' )
                stock_name = os.path.splitext(os.path.basename(root))[0]
                for file in files:
                    d = datetime.strptime(file.split('.')[0], '%Y-%m-%d').date()
                    # print(d)
                    stock_key = stock_name + ' ' + str(d)

                    date_tweets[stock_key] = list()
                    file_path = os.path.join(root, file)
                    #又读取一遍文本？？
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line_dict = json.loads(line)
                            text = line_dict['news_text']
                            word_idxes = self.get_word_idxes(text)
                            # print(word_idxes)
                            date_tweets[stock_key].append(word_idxes)
                            num_tweets += 1
        return date_tweets

    def get_word_idxes(self, words, expand=False, maxlen=None):
        '''
        :param words:传入text
        :param expand:默认F，if T 当word不在word2idx中时，在word2idx中添加word，idx为len(word2idx)
        if F word_idxes对应的位置填充为UNK
        :param maxlen:num 限制word_idx长度
        :return:word_idxes or real_len
        '''
        word_idxes = list()
        for w in words:
            if w in self.word2idx:
                word_idxes.append(self.word2idx[w])
            else:
                if expand:
                    word_idx = len(self.word2idx)
                    self.word2idx[w] = word_idx
                    self.idx2word[word_idx] = w
                    word_idxes.append(word_idx)
                else:
                    word_idxes.append(self.word2idx[self.UNK])

                self.num_UNK_words += 1
        self.num_words += len(words)

        if maxlen:
            real_len = len(word_idxes)
            if len(word_idxes) < maxlen:
                # padding
                while len(word_idxes) < maxlen:
                    word_idxes.append(self.word2idx[self.PAD])
            elif len(word_idxes) > maxlen:
                # slicing
                word_idxes = word_idxes[:maxlen]
            return word_idxes, real_len
        else:
            return word_idxes

    def load_stock_history(self):
        '''

        :return: stock_dict eg:'600028': [[datetime.date(2022, 1, 5), 0.002342]
        '''
        stock_dict = dict()
        diff_percentages = list()
        data_dir = r'C:\Users\46350\PycharmProjects\Algorithm\simulation\out_price'
        num_trading_days = 0
        file_names = os.listdir(data_dir)

        for filename in file_names:
            stock_name = os.path.splitext(os.path.basename(filename))[0]
            #print(stock_name)
            file_path = os.path.join(data_dir, filename)
            #print(file_path)

            with open(file_path, 'r', encoding='utf-8') as f:
                for l in list(f):
                    row = l.rstrip().split(' ')
                    stock_date = datetime.strptime(row[0], '%Y-%m-%d').date()
                    # # date filtering
                    # if not self.date_min <= stock_date <= self.date_max:
                    #     continue
                    # if not (date(2022, 1, 5) <= stock_date <= date(2022, 3, 15)):
                    #     continue

                    price_diff_percent = float(row[1])
                    if stock_name not in stock_dict:
                        stock_dict[stock_name] = list()
                    stock_dict[stock_name].append([stock_date, price_diff_percent])

                    num_trading_days += 1

                    if len(stock_dict[stock_name]) > 5:#???
                        diff_percentages.append(price_diff_percent)
        num_ex = 0
        for stock_name in stock_dict:
            num_ex += len(stock_dict[stock_name]) - 5
        return stock_dict

    def map_stocks_tweets(self):
        '''
        按时间来划分测试集训练集
        :return:train_x, train_y, dev_x, dev_y, test_x, test_y
        '''
        train_x = list()
        train_y = list()
        dev_x = list()
        dev_y = list()
        test_x = list()
        test_y = list()

        train_lable_freq_dict = dict()
        dev_lable_freq_dict = dict()
        test_lable_freq_dict = dict()

        diff_percentages = list()

        num_dates = 0
        num_tweets = 0
        zero_tweet_days = 0
        num_filtered_samples = 0

        for stock_name in self.stock_dict:

            stock_history = self.stock_dict[stock_name]
            #股票天数
            stock_days = len(stock_history)

            num_stock_dates = 0
            num_stock_tweets = 0
            stock_zero_tweet_days = 0

            for i in range(stock_days):
                stock_date = stock_history[i][0]

                ex = list()
                day_lens = list()
                news_lens = list()
                # found_tweet_days = 0

                days = list()

                num_empty_tweet_days = 0
                #时间倒退
                for j in [5, 4, 3, 2, 1]:
                    tweet_date = stock_date - timedelta(days=j)
                    stock_key = stock_name + ' ' + str(tweet_date)
                    # print(stock_key)
                    ex_1 = list()
                    t_lens = list()

                    if stock_key in self.date_tweets.keys():
                        tweets = self.date_tweets[stock_key]
                        # print(tweets)
                        #有问题
                        for w_idxes in tweets:
                            ex_1.append(' '.join([str(widx) for widx in w_idxes]))
                            # print(ex_1)
                            t_lens.append(len(w_idxes))
                            # print(t_lens)

                        day_lens.append(len(tweets))

                        num_stock_tweets += len(tweets)

                        if len(tweets) == 0:
                            num_empty_tweet_days += 1
                        else:
                            days.append(tweet_date)

                    else:
                        # no tweets date
                        day_lens.append(0)

                    ex.append('\n'.join(ex_1))
                    news_lens.append(t_lens)

                if num_empty_tweet_days > 0:
                    num_filtered_samples += 1
                    continue

                # StockNet
                #标注涨跌
                if stock_history[i][1] < 0.000:
                    label = 0
                elif 0.000 <= stock_history[i][1] <= 0.0065:
                    label = 1
                else:
                    label = 2

                label_date = stock_history[i][0]
                if date(2022, 2, 7) <= label_date < date(2022, 4,1):
                    train_x.append(ex)
                    train_y.append(label)

                    if label in train_lable_freq_dict.keys():
                        train_lable_freq_dict[label] += 1
                    else:
                        train_lable_freq_dict[label] = 1

                    num_dates += 5
                    num_stock_dates += 5

                elif date(2022, 4, 1) <= label_date < date(2022, 4, 14):
                    dev_x.append(ex)
                    dev_y.append(label)

                    if label in dev_lable_freq_dict:
                        dev_lable_freq_dict[label] += 1
                    else:
                        dev_lable_freq_dict[label] = 1

                    num_dates += 5
                    num_stock_dates += 5

                elif date(2022, 4, 14) <= label_date < date(2022, 4, 15):
                    test_x.append(ex)
                    test_y.append(label)

                    if label in test_lable_freq_dict.keys():
                        test_lable_freq_dict[label] += 1
                    else:
                        test_lable_freq_dict[label] = 1

                    num_dates += 5
                    num_stock_dates += 5
                else:
                    continue
                diff_percentages.append(stock_history[i][1])
        return train_x, train_y, dev_x, dev_y, test_x, test_y

    def _get_idxes_len(self, x_texts, max_date_len, max_news_len):
        x = list()  # (days, max_date_len, max_news_len)
        x_date_len = list()  # (days,)
        x_date_news_len = list()  # (days, max_date_len)

        # days
        for d in x_texts:
            news_word_idxes = list()
            news_lens = list()

            #d = d.decode()
            if len(d) > 0:
                news_list = d.split('\n')

                # slicing
                if len(news_list) > max_date_len:
                    news_list = news_list[:max_date_len]

                x_date_len.append(len(news_list))

                for n in news_list:
                    if len(n) > 0:
                        word_idxes = [int(w) for w in n.split(' ')]

                        # slicing
                        if len(word_idxes) > max_news_len:
                            word_idxes = word_idxes[:max_news_len]

                        news_lens.append(len(word_idxes))
                    else:
                        word_idxes = list()
                        news_lens.append(0)

                    if max_news_len is not None:
                        while len(word_idxes) < max_news_len:
                            word_idxes.append(self.word2idx[self.PAD])
                    news_word_idxes.append(word_idxes)
            else:
                x_date_len.append(0)  # pad

            if max_date_len is not None:
                while len(news_word_idxes) < max_date_len:
                    news_word_idxes.append(self.empty_news)  # pad

                while len(news_lens) < max_date_len:
                    news_lens.append(0)  # pad

            x.append(news_word_idxes)
            x_date_news_len.append(news_lens)

        x = np.array(x, dtype=np.int32)
        x_date_len = np.array(x_date_len, dtype=np.int32)
        x_date_news_len = np.array(x_date_news_len, dtype=np.int32)

        return x, x_date_len, x_date_news_len




def get_class_weights(n_samples, y, num_classes=3):
    return n_samples / (num_classes * np.bincount(y))