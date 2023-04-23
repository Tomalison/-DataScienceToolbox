import pandas as pd
url = 'https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv'
df = pd.read_csv(url)

# 1. Check if df has any missing values.(檢查資料是否有缺失值)
if df.isnull().values.any():
    print("The dataframe contains missing values")
else:
    print("The dataframe does not contain missing values")

# 2. Count the number of missing values in each column and find the maximum number of missing values.(計算每個欄位的缺失值數量，並找出最大的缺失值數量)
missing_count = df.isnull().sum()
max_missing = missing_count.max()
print(f"Number of missing values in each column:\n{missing_count}")
print(f"\nThe maximum number of missing values in a column is {max_missing}")

# 3. Replace missing values of multiple numeric columns with the mean.(取代多個數值欄位的缺失值為平均值)
numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
print(df)

# 4. Calculate the average price of different manufacturers.(計算不同製造商的平均價格)
average_price = df.groupby('Manufacturer')['Price'].mean()
print(f"\nAverage price of different manufacturers:\n{average_price}")

# 5. Replace missing values of price columns with the mean depending on its manufacturers.(取代價格欄位的缺失值為各製造商的平均值)
df['Price'] = df.groupby('Manufacturer')['Price'].transform(lambda x: x.fillna(x.mean()))
