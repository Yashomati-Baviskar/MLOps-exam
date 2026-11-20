from pathlib import Path
import pandas as pd

source = Path("data/raw/wine_v1.csv")
output = Path("data/raw/wine_raw.csv")

df = pd.read_csv(source)
df["collected_at"] = pd.Timestamp.now().isoformat()
output.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output, index=False)

print("Data collection completed")
print("Rows collected:", len(df))
print("Output:", output)
