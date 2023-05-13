import pandas as pd
import numpy as np

# 讀取資料集
df = pd.read_csv('https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/train.csv')

data = pd.DataFrame(df)

# 資料前處理
data.info()
data.describe()
data.isnull().sum()

# 將缺失值填補
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)
data['Cabin'].fillna('None', inplace=True)
data.isnull().sum()

# 新增特徵
data['Family_Size'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = 0
data.loc[data['Family_Size'] == 1, 'IsAlone'] = 1
data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
data['FareBin'] = pd.qcut(data['Fare'], 4)
data['AgeBin'] = pd.cut(data['Age'].astype(int), 5)

# 刪除不需要的欄位
data.drop(['Cabin'], axis=1, inplace=True)
data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
data.head()
data.info()
data.describe()
data.isnull().sum()

# 將資料集命名為 df_train，準備模型訓練
df_train = data
df_train.head()
df_train.info()
df_train.describe()
df_train.isnull().sum()

# 定義特徵欄位和目標欄位
# 使用 One-Hot Encoding 對字串特徵進行編碼
df_train_encoded = pd.get_dummies(df_train)

# 定義特徵欄位和目標欄位
columns_X = list(df_train_encoded.columns)
columns_X.remove('Survived')
columns_y = ['Survived']

train_X = df_train_encoded[columns_X]
train_y = df_train_encoded[columns_y]

# 使用 Logistic 迴歸模型進行交叉驗證
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

log = LogisticRegression(random_state=0, max_iter=3000)
scores = cross_val_score(log, train_X, train_y.values.ravel(), cv=5, scoring='accuracy')

print(scores)