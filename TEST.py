import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3

base_url = 'http://movies.yahoo.com.tw'
urls = ['/movie_thisweek.html', '/movie_intheaters.html', '/movie_comingsoon.html']

# 創建空的 DataFrame
df = pd.DataFrame(columns=['chinese_name', 'english_name', 'release_date', 'expectation'])

# 逐一處理每個分頁的數據
for url in urls:
    response = requests.get(base_url + url)
    content = response.content

    # 解析 HTML 內容內容
    soup = BeautifulSoup(content, 'html.parser')

    # 取得電影區塊的元素
    movie_section = soup.select_one('div.release_box')

    # 取得所有電影的資訊
    movies = movie_section.select('div.release_info_text')

    # 逐一解析電影信息並添加到 DataFrame
for movie in movies:
    # 取得中文名稱
    chinese_name = movie.select_one('div.release_movie_name a').text.strip()

    # 取得英文名稱
    english_name = movie.select_one('div.en').text.strip()

    # 取得上映日期
    release_date = movie.select_one('div.release_movie_time').text.strip()

    # 取得期待度
    expectation = movie.select_one('div.leveltext span').text.strip()

    # 添加電影信息到 DataFrame
    df = df.append({'chinese_name': chinese_name, 'english_name': english_name, 'release_date': release_date,
                    'expectation': expectation}, ignore_index=True)

# 將 DataFrame 存儲到 SQLite 數據庫
conn = sqlite3.connect('movie_info.db')  # 連結數據庫，如果不存在會自動創建
cursor = conn.cursor()

# 創建數據表
cursor.execute('''CREATE TABLE IF NOT EXISTS movie
              (id INTEGER PRIMARY KEY AUTOINCREMENT , 
              chinese_name TEXT,
              english_nameTEXT,
              release_date TEXT,
              expectation TEXT);''')

# 將 DataFrame 寫入數據庫
df.to_sql('movie', conn, if_exists='replace', index=False)

# 關閉數據庫連結
conn.close()