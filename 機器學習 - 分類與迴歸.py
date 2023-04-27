#練習：請根據給定的資料集，分成前 7 筆的訓練資料與後 3 筆的測試資料完成以下兩個分析：


利用 Country, Age, Purchased 對 Salary 進行迴歸學習，印出後三筆資料的 Salary 為何

import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

source = 'https://raw.githubusercontent.com/MachineLearningLiuMing/scikit-learn-primer-guide/master/Data.csv'

#讀取資料集
df = pd.read_csv( 'https://raw.githubusercontent.com/MachineLearningLiuMing/scikit-learn-primer-guide/master/Data.csv')

# 使用 SimpleImputer 填補缺失值，使用中位數填補
imputer = SimpleImputer(strategy='median')
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])

# 將 Country 欄位轉換成 one-hot encoding
df = pd.get_dummies(df, columns=['Country'])

#分割資料集為訓練資料和測試資料
train_df = df.iloc[:7 , :]
test_df = df.iloc[7: , :]

#將資料集拆分為特徵和標籤
X_train = train_df[['Age' , 'Salary','Country_France','Country_Germany','Country_Spain']]
y_train = train_df['Purchased']
X_test = test_df[['Age' , 'Salary','Country_France','Country_Germany' , 'Country_Spain']]

#建立logistic regression 模型
model = LogisticRegression(random_state=0)

#訓練模型
model.fit(X_train , y_train)

#預測測試資料集的Purchased
y_pred = model.predict(X_test)

#印出後三筆資料的 Purchased
print(y_pred)

#Task08-02-2
import pandas as pd
import statsmodels.api as sm

source = 'https://raw.githubusercontent.com/MachineLearningLiuMing/scikit-learn-primer-guide/master/Data.csv'

#讀取資料集
df = pd.read_csv(source)

# 使用 SimpleImputer 填補缺失值，使用中位數填補
imputer = SimpleImputer(strategy='median')
df[['Age','Salary']] = imputer.fit_transform(df[['Age','Salary']])
df['Purchased'] = df['Purchased'].replace({'No': 0, 'Yes': 1})
#將 Country 欄位轉換成 one-hot encoding
df = pd.get_dummies(df , columns = ['Country'])

#分割資料集為訓練資料和測試資料
train_df = df.iloc[:7 , :]
test_df = df.iloc[7: , :]

#將資料集拆分為特徵和標籤
X_train = train_df[['Country_France' , 'Country_Germany' , 'Country_Spain' , 'Age' , 'Purchased']]
y_train = train_df['Salary']

#建立線性回歸模型
model = sm.OLS(y_train , sm.add_constant(X_train))

#訓練模型
result = model.fit()

#印出模型參數
print(result.params)

#預測測試資料集的Salary
X_test = test_df[['Country_France', 'Country_Germany', 'Country_Spain', 'Age', 'Purchased']]
y_pred = result.predict(sm.add_constant(X_test))

#印出後三筆的 Salary
print(y_pred[-3:])
