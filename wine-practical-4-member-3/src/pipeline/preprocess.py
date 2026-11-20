from pathlib import Path
import pandas as pd

df = pd.read_csv("data/raw/wine_raw.csv")
df = df.drop_duplicates()

numeric_columns = df.select_dtypes(include="number").columns
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

df = df.dropna(subset=["wine_class"])
df = df.drop(columns=["collected_at"], errors="ignore")

Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_csv("data/processed/wine_preprocessed.csv", index=False)

print("Preprocessing completed")
print("Rows after preprocessing:", len(df))
