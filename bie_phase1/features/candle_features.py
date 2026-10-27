"""
فیچرهای Candle: Body, Upper Wick, Lower Wick, Body Ratio, Shadow Ratio
"""

import pandas as pd
import numpy as np


def compute_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    high_low_range = (df["high"] - df["low"]).replace(0, np.nan)

    out["body"] = (df["close"] - df["open"]).abs()
    out["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    out["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    out["body_ratio"] = out["body"] / high_low_range
    out["shadow_ratio"] = (out["upper_wick"] + out["lower_wick"]) / high_low_range
    return out
