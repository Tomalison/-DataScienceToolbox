import requests
from bs4 import BeautifulSoup
#匯入資料庫套件
import sqlite3
import pandas as pd

# 發送 GET 請求取得網頁內容
base_url = 'https://movies.yahoo.com.tw'
urls =[ '/movie_thisweek.html', '/movie_intheaters.html', '/movie_comingsoon.html']

#創建空的Dataframe
df = pd.DataFrame(columns=['movie_name', 'movie_name_en', 'movie_date', 'movie_expectation'])

#逐一處理每個分頁的數據
for url in urls:
    response = requests.get(base_url + url)
    content = response.content

    # 解析網頁內容
    soup = BeautifulSoup(content, 'html.parser')
    #取出電影區塊的資訊
    movies_section = soup.select_one('div.release_box')
    #取得所有電影的資訊
    movies = movies_section.select('div.release_info_text')

    #逐一解析電影信息並添加到 Dataframe
    for movie in movies:
        #取得中文名稱
        movie_name = movie.select_one('div.release_movie_name a').text.strip()
        #取得英文名稱
        movie_name_en = movie.select_one('div.en').text.strip()
        #取得上映日期
        movie_date = movie.select_one('div.release_movie_time').text.strip()
        #取得期待度
        movie_expectation = movie.select_one('div.leveltext span').text.strip()
        #將電影信息添加到 Dataframe
        df = pd.concat([df, pd.DataFrame({'movie_name': [movie_name], 'movie_name_en': [movie_name_en], 'movie_date': [movie_date], 'movie_expectation': [movie_expectation]})], ignore_index=True)
    #將Dataframe儲存到SQLite資料庫
    conn = sqlite3.connect('movies.db')
    cursor = conn.cursor()
    #創建數據表
    cursor.execute('''CREATE TABLE IF NOT EXISTS movies (id INTEGER PRIMARY KEY AUTOINCREMENT ,movie_name TEXT, movie_name_en TEXT, movie_date TEXT, movie_expectation TEXT);''')
    #將Dataframe中的數據寫入數據庫
    df.to_sql('movies', conn, if_exists='replace', index=False)
    #關閉資料庫連接
    conn.close()

    print(df)