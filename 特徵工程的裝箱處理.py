import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.random(100)*100)
n = input()

#裝箱法
df['box'] = pd.qcut(df[0], q = int(n), label=False)
print(df)