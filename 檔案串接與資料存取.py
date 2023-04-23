#練習：健保特約機構防疫家用快篩剩餘數量明細是因應疫情所釋出的開放資料，為了避免像之前那樣出現排隊搶購的情況，健保署規劃提供即時的庫存資訊給民眾參考，該資料包含醫事機構代碼、醫事機構名稱以及快篩試劑截至目前結餘存貨數量等。請嘗試用 Pandas 存取來自於檔案來源的資料，並且利用計算每個縣市的剩餘快篩總數並且排序剩餘數量前五高的資料。

#Sample Input: https://data.nhi.gov.tw/resource/Nhi_Fst/Fstdata.csv

#Sample Output: 剩餘快篩數量前五高的縣市

import pandas as pd
import numpy as np
#要先把醫事機構地址的欄位切割出來，再用groupby計算每個縣市的剩餘快篩總數

df = pd.read_csv('https://data.nhi.gov.tw/resource/Nhi_Fst/Fstdata.csv')
df['縣市'] = df['醫事機構地址'].str.split('市').str[0]
df['縣市'] = df['縣市'].str.split('縣').str[0]


df.groupby('縣市').agg({'快篩試劑截至目前結餘存貨數量': 'sum'}).sort_values(by='快篩試劑截至目前結餘存貨數量', ascending=False).head(5)
result = df.groupby('縣市').agg({'快篩試劑截至目前結餘存貨數量': 'sum'}).sort_values(by='快篩試劑截至目前結餘存貨數量', ascending=False).head(5)
print(result)


