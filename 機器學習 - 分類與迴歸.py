import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

source = 'https://raw.githubusercontent.com/MachineLearningLiuMing/scikit-learn-primer-guide/master/Data.csv'

df = pd.read_csv(source)

df['Purchased'] = df['Purchased'].map({'Yes': 1, 'No': 0})

# 將類別型資料轉換成數值型
df_ohe = pd.get_dummies(df, columns=['Country'])

# 補缺失值
df_ohe = df_ohe.fillna(df_ohe.mean())

# 分割資料
train_data = df_ohe.iloc[:7, :]
test_data = df_ohe.iloc[7:, :]

# 設定訓練資料和目標
X_train = train_data[['Country_France', 'Country_Germany', 'Country_Spain', 'Age', 'Salary']]
y_train = train_data['Purchased']

X_test = test_data[['Country_France', 'Country_Germany', 'Country_Spain', 'Age', 'Salary']]
y_test = test_data['Purchased']

# 創建分類樹模型
clf_model = DecisionTreeClassifier()

# 訓練模型
clf_model.fit(X_train, y_train)

# 使用模型進行預測
y_pred_clf = clf_model.predict(X_test)
print(y_pred_clf)
accuracy = accuracy_score(y_test, y_pred_clf)
print("Accuracy:", accuracy)

# 利用 Country, Age, Purchased 對 Salary 進行迴歸學習，印出後三筆資料的 Salary 為何
# 創建線性回歸模型
# 設定訓練資料和目標
X_train = train_data[['Country_France', 'Country_Germany', 'Country_Spain', 'Age', 'Purchased']]
y_train = train_data['Salary']

X_test = test_data[['Country_France', 'Country_Germany', 'Country_Spain', 'Age', 'Purchased']]
y_test = test_data['Salary']
model = LinearRegression()
# 訓練模型
model.fit(X_train, y_train)
# 使用模型進行預測
y_pred = model.predict(X_test)
print(y_pred)