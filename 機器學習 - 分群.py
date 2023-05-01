import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

print('===== 原始資料 =====')
df = pd.DataFrame(X)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_
print('分群結果3:', labels)

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
labels = kmeans.labels_
print('分群結果4:', labels)

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
labels = kmeans.labels_
print('分群結果5:', labels)


# K=3時比較好，因為當K=3時可以很明確的分為三個群體 k=4、5時，分布並不平均