import pandas as pd

df = pd.read_csv("data/processed/wine_features.csv")
required_columns = [
    "alcohol", "malic_acid", "ash", "total_phenols",
    "flavanoids", "color_intensity", "proline", "target",
    "wine_class", "phenols_flavanoids_ratio",
    "color_proline_ratio", "alcohol_band"
]

missing_columns = [c for c in required_columns if c not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")
if df.empty:
    raise ValueError("Dataset is empty")
if df[required_columns].isnull().any().any():
    raise ValueError("Missing values found in required columns")

print("Data validation successful")
print("Dataset shape:", df.shape)
print("All required columns are present")
print("No missing values found")
