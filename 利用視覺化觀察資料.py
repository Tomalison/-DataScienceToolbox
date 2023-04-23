import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
df = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                     columns = iris['feature_names'] + ['target'])

# 繪製線圖
sns.lineplot(data=df[['sepal length (cm)', 'sepal width (cm)', \
                      'petal length (cm)', 'petal width (cm)']])
plt.show()

# 計算相關係數
correlation_matrix = df[['sepal length (cm)', 'sepal width (cm)', \
                         'petal length (cm)', 'petal width (cm)']].corr()
print(correlation_matrix)