import pandas as pd
df = pd.read_csv("data/processed/clean.csv")
print(df.isnull().sum())
print(df[["cEXT","cNEU","cAGR","cCON","cOPN"]].value_counts().head())
