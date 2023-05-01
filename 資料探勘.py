#練習：請根據給定的資料集，計算出「Frequent Patterns（Apriori）」和「Association Rules」的結果。

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

dataset = [
  ['Milk', 'Onion', 'Nutmeg', 'Kidney' 'Beans', 'Eggs', 'Yogurt'],
  ['Dill', 'Onion', 'Nutmeg', 'Kidney' 'Beans', 'Eggs', 'Yogurt'],
  ['Milk', 'Apple', 'Kidney' 'Beans', 'Eggs'],
  ['Milk', 'Unicorn', 'Corn', 'Kidney' 'Beans', 'Yogurt'],
  ['Corn', 'Onion', 'Onion', 'Kidney' 'Beans', 'Ice cream', 'Eggs'],
]
# 將資料轉換成0和1的格式
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
#print(df)
# 計算出頻繁項目集
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
#print(frequent_itemsets)

# 計算出關聯規則
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print(rules)