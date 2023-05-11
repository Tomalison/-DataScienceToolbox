'''在資料分析的工作流程中，以下是一些必備的環節，即使不考慮準確度：
定義問題：明確確定分析的目標和問題，確保理解業務需求和期望結果。
資料收集：收集相關的數據和資料，可以從多個來源獲取，包括數據庫、文件、API、網站等。
數據清理：對收集到的數據進行清理和預處理，包括處理缺失值、處理異常值、處理重複值、標準化數據格式等。
探索性數據分析（EDA）：通過統計和視覺化方法探索數據，理解數據的分佈、相關性和特徵。
特徵選擇和特徵工程：根據目標變量和特徵之間的關係，選擇重要的特徵或創建新的特徵，以提高模型的性能。
建模和分析：選擇合適的機器學習或統計模型來分析數據，並進行訓練和評估模型的性能。
結果解釋和報告：解釋和解讀模型的結果，向利益相關者提供清晰且可理解的報告，以便他們可以做出相應的決策。
部署和監控：將模型部署到實際環境中，並定期監控模型的性能和準確度，以確保模型的有效運作。
以上是在不考慮準確度情況下資料分析工作流程中的必備環節。準確度是一個重要的評估指標，並在模型選擇和評估中發揮關鍵作用。'''

# Path: 鐵達尼號的存活預測Baseline版本.py
#在不考慮準確度的前提下完成 Baseline ，讓已讀入的資料完成必要的資料前處理的操作。（補充：將 df 變成乾淨的 df_train，讓原始資料變成是模型跑得動的資料格式。）
import pandas as pd
import numpy as np
df = pd.read_csv('https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/train.csv')

data = pd.DataFrame(df)
#print(data)
data.info()
data.describe()
data.isnull().sum()
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)
data['Cabin'].fillna('None', inplace=True)
data.isnull().sum()
data['Family_Size'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = 0
data.loc[data['Family_Size'] == 1, 'IsAlone'] = 1
data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
data['FareBin'] = pd.qcut(data['Fare'], 4)
data['AgeBin'] = pd.cut(data['Age'].astype(int), 5)
#刪除遺失值最多的欄位
data.drop(['Cabin'], axis=1, inplace=True)

#刪除不需要的欄位
data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
data.head()
data.info()
data.describe()
data.isnull().sum()
# One-hot encoding
data = pd.get_dummies(data)
data.head()
data.info()
data.describe()
data.isnull().sum()
#將資料分為訓練資料與測試資料
train = data[:len(df)]
test = data[len(df):]
test.drop(['Survived'], axis=1, inplace=True)
#將訓練資料分為訓練資料與驗證資料
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train.drop(['Survived'], axis=1), train['Survived'], test_size=0.2, random_state=2020)
#建立模型
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)
#預測驗證資料
y_pred = model.predict(X_val)
#評估模型
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' %accuracy_score(y_val, y_pred))
#預測測試資料
y_pred = model.predict(test)
#輸出預測結果








