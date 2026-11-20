import pandas as pd

df = pd.read_csv("data/processed/wine_preprocessed.csv")
df["phenols_flavanoids_ratio"] = (
    df["total_phenols"] / df["flavanoids"].replace(0, 1)
)
df["color_proline_ratio"] = (
    df["color_intensity"] / df["proline"].replace(0, 1)
)
df["alcohol_band"] = pd.cut(
    df["alcohol"],
    bins=3,
    labels=["low", "medium", "high"]
).astype(str)

df.to_csv("data/processed/wine_features.csv", index=False)
print("Feature engineering completed")
print("New features: phenols_flavanoids_ratio, color_proline_ratio, alcohol_band")
