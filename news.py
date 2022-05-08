
from urllib.request import urlopen

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from urllib import parse
import requests
import re
data = pd.DataFrame(columns=['date', 'href', 'title'])
href = []
title = []
date = []
news_text = []
for i in range(40):
    url = "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php?symbol=sh601857&Page=%d"%(i + 1)
    html = urlopen(url)  # 打开所需爬取的页面
    bs = BeautifulSoup(html, 'html.parser', from_encoding="GBK")  # 用BeautifulSoup解析网页

    p1 = bs.findAll('div', {'class': 'datelist'})  # 找到新闻标题的所在标签名称

    for each in p1:
        x = each.select('a')
        all_date = re.findall('(\d{4}-\d{2}-\d{2})', each.ul.text)
        #y = each.select('')

        for i in range(len(x)):
            page_url = x[i]['href']
            result1 = requests.get(page_url)  # 进入新闻链接
            result1.encoding = 'utf-8'  # 让中文可以正常显示
            content1 = result1.content
            soup1 = BeautifulSoup(content1, 'html.parser', from_encoding=result1.encoding)  #
            p2 = soup1.findAll('div', {'class': 'article'})
            for each in p2:
                a = each.select('p')
                content = []
                for i2 in range(len(a)):
                    content.append(re.sub('\s', '', a[i2].text))
                article = ''.join(content)
                news_text.append(article)
            if len(p2) == 0:
                p3 = soup1.findAll('div', {'class': 'blk_container'})
                for each in p3:
                    a = each.select('p')
                    content = []
                    for i3 in range(len(a)):
                        content.append(re.sub('\s', '', a[i3].text))
                    article = ''.join(content)
                    news_text.append(article)
                #news_text.append(np.NaN)
                if len(p3) == 0:
                    news_text.append(np.NAN)
            href.append(x[i]['href']) # 即a标签下的title
            title.append(x[i].text)
            date.append(all_date[i])
href = pd.Series(href, dtype=object)
title = pd.Series(title, dtype=object)
date = pd.Series(date, dtype=object)
news_text = pd.Series(news_text, dtype=object)
one_page_data = pd.concat([date, href, title, news_text], axis=1)
one_page_data.columns = ['date', 'href', 'title', 'news_text']
one_page_data.to_csv('sh601857.csv')
one_page_data = pd.read_csv(r'C:\Users\46350\PycharmProjects\Algorithm\sh601857.csv')
#data = pd.concat([data, one_page_data])
#data.head()
one_page_data.index = pd.to_datetime(one_page_data['date'])
one_page_data.head()
#one_page_data.index

from datetime import datetime, timedelta, date
start_date = datetime.strptime(one_page_data['date'][0], "%Y-%m-%d")
end_date = datetime.strptime(one_page_data['date'][-1], "%Y-%m-%d")
date_time = start_date
while date_time >= end_date:
    if date_time in one_page_data.index:
        day_news = one_page_data.loc[date_time]
        day_news.to_csv('simulation/sh601857/' + str(datetime.date(date_time))+'.csv')
    date_time -= timedelta(days=1)
# day_news = one_page_data.loc[date_time]
# date(2022, 3, 15)

import jqdatasdk
jqdatasdk.auth("13580430532", "Q1230321q")
#按分钟获取数据
result = jqdatasdk.get_price("601857.XSHG", start_date="2022-02-01", end_date="2022-04-12 23:59:00",frequency='daily')
result.index
Rise_Percent = []
for i in range(len(result)-1):
    rise = (result['open'][i+1] - result['open'][i])/result['open'][i]
    Rise_Percent.append(rise)
index = result.index
date = pd.Series(index, name='date')
Rise = pd.Series(Rise_Percent, name='rise_per')
rise_date = pd.concat([date, Rise], axis=1)
#输出到csv文件中
rise_date.to_csv('601857.txt', sep=' ', encoding='utf-8', index=False)