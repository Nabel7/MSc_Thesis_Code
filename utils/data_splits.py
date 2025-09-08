# utils/data_splits.py
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict


RAW_CSV = Path(r"F:\MSc_Thesis\MSc_Thesis_Code\data\speculator_per_side.csv")
OUT_DIR = Path(r"F:\MSc_Thesis\MSc_Thesis_Code\data")
OUT_DIR.mkdir(exist_ok=True)

# Load & sort
df = pd.read_csv(RAW_CSV, parse_dates=["DeliveryPeriod"])
df = df.sort_values("DeliveryPeriod").reset_index(drop=True)

# Split sizes
n = len(df)
n_train = int(n * 0.7)
n_val   = int(n * 0.85)   # 70% train, 15% val, 15% test

# Save splits
df.iloc[:n_train].to_csv(OUT_DIR / "speculator_train.csv", index=False)
df.iloc[n_train:n_val].to_csv(OUT_DIR / "speculator_val.csv", index=False)
df.iloc[n_val:].to_csv(OUT_DIR / "speculator_test.csv", index=False)

print("Splits written to", OUT_DIR)
print(f"Train: {n_train}, Val: {n_val-n_train}, Test: {n-n_val}")


def compute_norm_stats(df: pd.DataFrame, feature_cols) -> Dict[str, np.ndarray]:
    """Fit mean/std on df[feature_cols]. Replace 0 std with 1."""
    X = df[feature_cols].astype(np.float32)
    mu = X.mean().astype(np.float32)
    sd = X.std().replace(0, 1).astype(np.float32)
    return {"mu": mu.to_dict(), "sd": sd.to_dict()}
