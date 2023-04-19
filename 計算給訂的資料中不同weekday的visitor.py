#請利用 Pandas 計算給定的資料中不同 weekday 的 visitor 總和為何？

#Sample Output: { 'Sun': 376, 'Mon': 782 }

# Sample Code：

import pandas as pd
import numpy as np

d = [
    {'city': 'Austin', 'visitor': 139, 'weekday': 'Sun'},
    {'city': 'Dallas', 'visitor': 237, 'weekday': 'Sun'},
    {'city': 'Austin', 'visitor': 326, 'weekday': 'Mon'},
    {'city': 'Dallas', 'visitor': 456, 'weekday': 'Mon'}
]
#請利用 Pandas 計算給定的資料中不同 weekday 的 visitor 總和為何？
df = pd.DataFrame(d)
df.groupby('weekday').agg({'visitor': 'sum'})
result = df.groupby('weekday').agg({'visitor': 'sum'})
print(result)

