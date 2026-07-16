import pandas as pd

df = pd.read_csv("data/bbc-text.csv")

print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())