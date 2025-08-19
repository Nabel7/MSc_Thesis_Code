# utils/data_splits.py
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict

def slice_by_dates(df: pd.DataFrame,
                   start: Optional[str] = None,
                   end: Optional[str]   = None) -> pd.DataFrame:
    """Return a view filtered to [start, end] inclusive by DeliveryPeriod."""
    m = pd.Series(True, index=df.index)
    if start:
        m &= (df["DeliveryPeriod"] >= pd.to_datetime(start))
    if end:
        m &= (df["DeliveryPeriod"] <= pd.to_datetime(end))
    return df.loc[m]

def compute_norm_stats(df: pd.DataFrame, feature_cols) -> Dict[str, np.ndarray]:
    """Fit mean/std on df[feature_cols]. Replace 0 std with 1."""
    X = df[feature_cols].astype(np.float32)
    mu = X.mean().astype(np.float32)
    sd = X.std().replace(0, 1).astype(np.float32)
    return {"mu": mu.to_dict(), "sd": sd.to_dict()}
