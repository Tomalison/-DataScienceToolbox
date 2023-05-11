#載入資料與認識資料
import pandas as pd
import numpy as np

#載入 Titanic 資料集的 `train.csv` 資料集

#（資料網址：https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/train.csv）
df = pd.read_csv('https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/train.csv')


#根據 Kaggle 文件瞭解 Titanic 資料中所有欄位的定義為何？
'''PassengerId乘客編號,Survived倖存狀況,Pclass類 ,Name姓名,Sex性別,Age年齡,SibSp同遊親屬數,Parch直系親屬數,Ticket船票號碼,Fare票價,Cabin客艙編號,Embarked登船口岸'''

#進一步觀察每一個欄位「數值欄位/類別欄位」與「是否生存欄位」之關係程度，找出關係大的欄位。
#（Hint: 可以利用視覺化圖表、 correlation、feature importance 等方法）
import matplotlib.pyplot as plt
import seaborn as sns
#數值欄位
df_num = df[['Survived','Pclass','Age','SibSp','Parch','Fare']]
#類別欄位
df_class = df[['PassengerId','Survived','Name','Ticket','Cabin','Embarked']]
#數值欄位與是否生存欄位關係
g = sns.heatmap(df_num.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()
#類別欄位與是否生存欄位關係
#轉換類別欄位為數值
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df_class['Name'] = labelencoder.fit_transform(df_class['Name'])
df_class['Ticket'] = labelencoder.fit_transform(df_class['Ticket'].astype(str))
df_class['Cabin'] = labelencoder.fit_transform(df_class['Cabin'].astype(str))
df_class['Embarked'] = labelencoder.fit_transform(df_class['Embarked'].astype(str))
df_class['PassengerId'] = labelencoder.fit_transform(df_class['PassengerId'].astype(str))
g = sns.heatmap(df_class.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()

#承上題，請問你是如何定義與解讀什麼稱為「關係大」呢？
'''數值欄位與是否生存欄位關係:Survived與數值的相關係數絕對值最大為0.26，數值欄位與是否生存欄位關係不夠大
    類別欄位與是否生存欄位關係:Survived與類別的相關係數絕對值最大為0.25(客艙)，類別欄位與是否生存欄位關係不夠大'''
