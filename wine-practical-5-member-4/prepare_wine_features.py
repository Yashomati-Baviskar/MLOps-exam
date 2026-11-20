from pathlib import Path
import pandas as pd
from sklearn.datasets import load_wine

# Load wine dataset
wine = load_wine(as_frame=True)

df = wine.frame.copy()

# Add wine class names
df["wine_class"] = df["target"].map(
    dict(enumerate(wine.target_names))
)

# Add unique sample IDs
df["sample_id"] = range(1, len(df) + 1)

# Add timestamp
df["event_timestamp"] = pd.Timestamp.now(tz="UTC").floor("s")

# Feature engineering
df["phenols_flavanoids_ratio"] = (
    df["total_phenols"] / df["flavanoids"].replace(0, 1)
)

df["color_proline_ratio"] = (
    df["color_intensity"] / df["proline"].replace(0, 1)
)

# Rename column
df = df.rename(columns={
    "od280/od315_of_diluted_wines": "od280_od315_diluted_wines"
})

# Create data folder
Path("data").mkdir(exist_ok=True)

# Save dataset
df.to_parquet("data/wine_features.parquet", index=False)

print("Wine feature data created successfully!")
print("Dataset shape:", df.shape)
print("Saved to data/wine_features.parquet")