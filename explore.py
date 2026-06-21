import pandas as pd


df = pd.read_csv("oasis_longitudinal.csv")
print(df.shape)
print(df.head())
print(df.columns.tolist())
print(df.isnull().sum())
print(df["Group"].value_counts())
