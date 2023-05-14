'''變量名	中文譯名	詳情
PassengerId	乘客編號	無意義，僅用於為乘客進行編號，從1-1309
survival	是否倖存
本次預測項目的目標——TestSet將其抹去，並作為評分標準。

在歷史上，這個乘客有沒有倖存下來。如果活下來標為1，沒有存活則標為0.

pclass	船艙等級	1=頭等艙； 2=二等艙； 3=三等艙（Kaggle將其作為乘客社會地位的象徵）
sex	乘客性別	標為male與female。
name	姓名	乘客的全名。emmmmm，這個特徵一來不太好用，二來適合用來作弊.......
Age	年齡
乘客的年齡，有很多NaN（其實在網上能夠找到......)

sibsp	同輩親屬
按照Kaggle的說法，這是sibling（兄弟姐妹）、spouse（配偶）的合寫。

對於某個乘客而言，如果船上有他們的sibling或者spouse，那麼有幾個sibsp就是幾，parch下同。

”Sibling“定義為”兄弟姐妹、義兄弟(stepbrother)、義姐妹（stepsister)”；

 "Spouse"僅限於正式夫妻。情人、未婚夫妻（mistresses, fiances)不計入

parch	長輩晚輩
”parch“是”parent“和”child“的縮寫。僅限於父母、子女、繼子女。

Kaggle特別指出，有些孩子由保姆跟隨旅行，他們的parch為0

ticket	船票編號	船票的編號，編號的形式有一點奇特，有的是純數字，有的帶有一些英文字母
fare	票價	乘客購票的價格
cabin	客艙號	乘客當時所處的客艙號碼，有少量的數據（大多數的NaN）
Embarked	登船港口
C = Cherbourg（法國瑟堡），S = Southampton（英國南安普頓），

Q = Queenstown（ 昆士敦，今稱科夫Cobh，位於愛爾蘭），有少量的NaN'''
'''
數據處理
　　除了因變量survived之外，Kaggle給出了十種特徵。

　　在這次臨時起意中，sibsp/parch/pclass這三個離散變量被完整保留。而name/cabin/ticket由於”意義不明“被我直接切除，換句話說，在這次分析過程中，我只採用了7種特徵。

　　sex - female 標為0， male標為1
　　age - 缺失值採用平均值填補（這裡是一處敗筆，後來复盤的時候覺得這裡應該考慮進行回歸，並考慮年齡缺失這一現像是否與乘客倖存與否有關），並進行Z-Score做標準化。
　　fare - 用Z-score做標準化
　　embarked - 一分為三，做成三個0-1變量Cherbourg/Southampton/Queenstown，刪除embarked列。其中Southampton的比例為70%以上（歷史上Titanic航行的出發地）。'''

#載入資料與認識資料
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#載入 Titanic 資料集的 `train.csv` 資料集

#（資料網址：https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/train.csv）
df = pd.read_csv('https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/train.csv')

#觀察資料集的資訊
#print(df.info())
#用表格完整觀察資料集的內容
#print(df)
#print(df.shape)
#統計描述資料集

df['Age'] = df['Age'].fillna(df['Age'].mean())
#print(df.describe())

#觀察性別是否影響生存
'''df['Died'] = 1 - df['Survived']
df.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7),stacked=True, color=['g', 'r'])
plt.show()'''

#再結合年齡是否會影響生存
'''fig = plt.figure(figsize=(25, 7))
sns.violinplot(x='Sex', y='Age',
               hue='Survived', data=df,
               split=True,
               palette={0: "r", 1: "g"}
              )
plt.show()'''
#觀察票價是否影響生存
'''figure = plt.figure(figsize=(25, 7))
plt.hist([df[df['Survived'] == 1]['Fare'], df[df['Survived'] == 0]['Fare']],
            stacked=True,color = ['g','r'],bins = 50, edgecolor='black',label = ['Survived','Dead'])

plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
plt.show()'''

#將年齡與票價結合再一起觀察生存
'''plt.figure(figsize=(25, 7))
ax = plt.subplot()
ax = plt.scatter(df[df['Survived'] == 1]['Age'], df[df['Survived'] == 1]['Fare'], c='green', s=df[df['Survived'] == 1]['Fare'])
ax = plt.scatter(df[df['Survived'] == 0]['Age'], df[df['Survived'] == 0]['Fare'], c='red', s=df[df['Survived'] == 0]['Fare'])
plt.show()'''

#觀察艙等是否影響生存
'''plt.figure(figsize=(25, 7))
sns.violinplot(x='Pclass', y='Fare', palette=['r', 'g'], hue='Survived', data=df, split=True)
plt.show()'''

#觀察登船港口是否影響生存
'''sns.barplot(x='Embarked', y='Survived', data=df)
plt.show()'''

def status(feature):
    print('Processing', feature, ': ok')

#將訓練資料與測試資料合併
def get_combined_data():
    # reading train data
    train = pd.read_csv('https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/train.csv')

    # reading test data
    test = pd.read_csv('https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/test.csv')

    # extracting and then removing the targets from the training data
    targets = train.Survived
    train.drop('Survived', axis = 1, inplace=True)

    # merging train data and test data for future feature engineering
    # we'll also remove the PassengerID since this is not an informative feature
    combined = pd.concat([train, test], ignore_index=True)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)

    return combined
combined = get_combined_data()
#print(combined.shape)

#找尋名字中的稱謂
titles = set()
for name in df['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())
#print(titles)
def get_titles():
    global combined

    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"
                        }

    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)
get_titles()
#print(combined.head())
combined[combined['Title'].isnull()]
#print(combined.iloc[1305])
#將稱謂轉換成數字
def process_names():
    global combined
    # we clean the Name variable
    combined.drop('Name', axis=1, inplace=True)

    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, titles_dummies], axis=1)

    # removing the title variable
    combined.drop('Title', axis=1, inplace=True)
process_names()
#print(combined.head())

#print(combined.iloc[:891].Age.isnull().sum())
#print(combined.iloc[891:].Age.isnull().sum())
print(combined.columns)

grouped_train = combined.iloc[:891].groupby(['Sex','Pclass','Title'])
grouped_median_train = grouped_train.median()
grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
print(grouped_median_train.head())

def fill_age(row):
    condition = (
        (grouped_median_train['Sex'] == row['Sex']) &
        (grouped_median_train['Title'] == row['Title']) &
        (grouped_median_train['Pclass'] == row['Pclass'])
    )
    return grouped_median_train[condition]['Age'].values[0]


def process_age():
    global combined
    # a function that fills the missing values of the Age variable
    combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
    status('age')
    return combined

combined = process_age()

def process_names():
    global combined
    # we clean the Name variable
    combined.drop('Name', axis=1, inplace=True)

    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, titles_dummies], axis=1)

    # removing the title variable
    combined.drop('Title', axis=1, inplace=True)

    status('names')
    return combined
combined = process_names()

def process_fares():
    global combined
    # there's one missing fare value - replacing it with the mean.
    combined.Fare.fillna(combined.iloc[:891].Fare.mean(), inplace=True)
    status('fare')
    return combined
combined = process_fares()

def process_embarked():
    global combined
    # two missing embarked values - filling them with the most frequent one in the train  set(S)
    combined.Embarked.fillna('S', inplace=True)
    # dummy encoding
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop('Embarked', axis=1, inplace=True)
    status('embarked')
    return combined
combined = process_embarked()
combined.head()

train_cabin, test_cabin = set(), set()

for c in combined.iloc[:891]['Cabin']:
    try:
        train_cabin.add(c[0])
    except:
        train_cabin.add('U')

for c in combined.iloc[891:]['Cabin']:
    try:
        test_cabin.add(c[0])
    except:
        test_cabin.add('U')

print train_cabin
print test_cabin

def process_cabin():
    global combined
    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U', inplace=True)

    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])

    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')
    combined = pd.concat([combined, cabin_dummies], axis=1)

    combined.drop('Cabin', axis=1, inplace=True)
    status('cabin')
    return combined
combined = process_cabin()

combined.head()

def process_sex():
    global combined
    # mapping string values to numerical one
    combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})
    status('Sex')
    return combined
combined = process_sex()

def process_pclass():

    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")

    # adding dummy variable
    combined = pd.concat([combined, pclass_dummies],axis=1)

    # removing "Pclass"
    combined.drop('Pclass',axis=1,inplace=True)

    status('Pclass')
    return combined
combined = process_pclass()

def cleanTicket(ticket):
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = map(lambda t : t.strip(), ticket)
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else:
        return 'XXX'

    tickets = set()
    for t in combined['Ticket']:
        tickets.add(cleanTicket(t))

        print len(tickets)

    def process_ticket():

        global combined

        # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
        def cleanTicket(ticket):
            ticket = ticket.replace('.', '')
            ticket = ticket.replace('/', '')
            ticket = ticket.split()
            ticket = map(lambda t: t.strip(), ticket)
            ticket = filter(lambda t: not t.isdigit(), ticket)
            if len(ticket) > 0:
                return ticket[0]
            else:
                return 'XXX'

    def process_family():

        global combined
        # introducing a new feature : the size of families (including the passenger)
        combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

        # introducing other features based on the family size
        combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
        combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
        combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

        status('family')
        return combined