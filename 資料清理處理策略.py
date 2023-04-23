#請利用題目提供的資料，分別完成以下幾種策略的資料填補方法：

#Drop missing observations
#Drop rows where all cells in that row is NA
#Create a new column full of missing values
#Fill in missing data with zeros
#Fill in missing in preTestScore with the mean value of preTestScore
#Fill in missing in postTestScore with each sex’s mean value of postTestScore
#Select some rows but ignore the missing data points

import numpy as np
import pandas as pd
raw_data = {'first_name': ['Jason', np.nan, 'Tina', 'Jake', 'Amy'],
        'last_name': ['Miller', np.nan, 'Ali', 'Milner', 'Cooze'],
        'age': [42, np.nan, 36, 24, 73],
        'sex': ['m', np.nan, 'f', 'm', 'f'],
        'preTestScore': [4, np.nan, np.nan, 2, 3],
        'postTestScore': [25, np.nan, np.nan, 62, 70]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'sex', 'preTestScore', 'postTestScore'])
#Drop missing observations(刪除缺失的觀測值)
df_drop = df.dropna()
#print(df_drop)

#Drop rows where all cells in that row is NA(刪除所有值均為缺失的觀測值)
df_drop_all = df.dropna(how='all')
#print(df_drop_all)

#Create a new column full of missing values(建立填滿缺失值的新欄位)
df_fill = df.copy()
df_fill['new_column'] = np.nan
#print(df_fill)

#Fill in missing data with zeros(填補缺失值為0)
df_fill_zero = df.fillna(0)
#print(df_fill_zero)

#Fill in missing in preTestScore with the mean value of preTestScore(填補preTestScore的缺失值為平均值)
df_fill_preTestScore = df.fillna({'preTestScore': df['preTestScore'].mean()})
#print(df_fill_preTestScore)

#Fill in missing in postTestScore with each sex’s mean value of postTestScore(填補postTestScore的缺失值為各性別的平均值)
df_fill_preTestScore_sex = df.copy()
df_fill_preTestScore_sex['preTestScore'] = df_fill_preTestScore_sex.groupby('sex')['preTestScore'].transform(lambda x: x.fillna(x.mean()))
#print(df_fill_preTestScore_sex)

#Select some rows but ignore the missing data points(選擇一些觀測值但忽略缺失的資料點)
df_select = df.loc[df['age'].notnull()]
#print(df_select)

#print(df)
