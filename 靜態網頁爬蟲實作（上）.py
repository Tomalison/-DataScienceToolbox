import requests
from bs4 import BeautifulSoup

# 發送 GET 請求取得網頁內容
url = 'https://movies.yahoo.com.tw/movie_thisweek.html'
response = requests.get(url)

# 使用 BeautifulSoup 解析網頁內容
soup = BeautifulSoup(response.text, 'html.parser')

# 找出目標元素，並提取資訊
movies = soup.select('ul.release_list li')
for movie in movies:
 # 中文名稱
 chinese_name = movie.select_one('div.release_movie_name a').text.strip()
 # 英文名稱
 english_name = movie.select_one('div.release_movie_name div.en').text.strip()
 # 上映日期
 release_date = movie.select_one('div.release_movie_time').text.strip()
 # 期待度
 expectation = movie.select_one('div.leveltext span').text.strip()

 print('中文名稱：', chinese_name)
 print('英文名稱：', english_name)
 print('上映日期：', release_date)
 print('期待度：', expectation)
 print('--------------------------')