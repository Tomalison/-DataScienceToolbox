import requests
from bs4 import BeautifulSoup

# 發送 GET 請求取得網頁內容
base_url = 'https://movies.yahoo.com.tw'
urls =[ '/movie_thisweek.html', '/movie_intheaters.html', '/movie_comingsoon.html']

for url in urls:
    response = requests.get(base_url + url)
    content = response.content

      # 解析網頁內容
    soup = BeautifulSoup(content, 'html.parser')
    #取出電影區塊的資訊
    movies_section = soup.select_one('div.release_box')
    #取得所有電影的資訊
    movies = movies_section.select('div.release_info_text')
    #print(movies)
    #取得所有電影的名稱
    for movie in movies:
        #取得中文名稱
        movie_name = movie.select_one('div.release_movie_name a').text.strip()
        #取得英文名稱
        movie_name_en = movie.select_one('div.en').text.strip()
        #取得上映日期
        movie_date = movie.select_one('div.release_movie_time').text.strip()
        #取得期待度
        movie_expectation = movie.select_one('div.leveltext span').text.strip()
#輸入電影資訊
        print('中文名稱：', movie_name)
        print('英文名稱：', movie_name_en)
        print('上映日期：', movie_date)
        print('期待度：', movie_expectation)
        print('-------------------')

