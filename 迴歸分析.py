import pandas as pd
import statsmodels.api as sm

source = 'https://raw.githubusercontent.com/cbrownley/foundations-for-analytics-with-python/master/statistics/winequality-both.csv'
df = pd.read_csv(source)
#將 `winequality-both` 資料的後 10 筆資料當成「測試資料」、其餘資料作為「訓練資料」；想要利用除了 type、quality 之外的欄位（X） 對 quality 欄位（y） 進行迴歸分析。
#請嘗試用 statsmodels 建立迴歸模型，並利用測試資料評估模型的準確度。
#提示：請參考 https://www.statsmodels.org/stable/regression.html#regression

#將後10 筆資料當成「測試資料」
test = df.tail(10)
testy = test['quality']
testX = sm.add_constant(test[test.columns.difference(['type','quality'])])

#將除了type、quality之外的欄位作為訓練資料
train = df.head(-10)
y = train['quality']
X = sm.add_constant(train[train.columns.difference(['type','quality'])])
#使用最小平方法進行迴歸分析
model = sm.OLS(y, X).fit()
#印出模型摘要
print(model.summary())
#預測測試資料的quality值
print(model.predict(testX))