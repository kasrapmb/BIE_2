"""
فیچرهای Volume: RVOL, OBV, CMF, MFI, Dollar Volume
"""

import pandas as pd
import numpy as np


def compute_rvol(volume: pd.Series, window: int = 20) -> pd.Series:
    avg_volume = volume.rolling(window).mean()
    return volume / avg_volume.replace(0, np.nan)


def compute_obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    obv = (direction * df["volume"]).cumsum()
    return obv


def compute_cmf(df: pd.DataFrame, window: int = 20) -> pd.Series:
    high_low_range = (df["high"] - df["low"]).replace(0, np.nan)
    money_flow_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / high_low_range
    money_flow_volume = money_flow_multiplier * df["volume"]
    cmf = money_flow_volume.rolling(window).sum() / df["volume"].rolling(window).sum().replace(0, np.nan)
    return cmf


def compute_mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = typical_price * df["volume"]

    price_diff = typical_price.diff()
    positive_flow = money_flow.where(price_diff > 0, 0.0)
    negative_flow = money_flow.where(price_diff < 0, 0.0)

    positive_sum = positive_flow.rolling(window).sum()
    negative_sum = negative_flow.rolling(window).sum()

    money_ratio = positive_sum / negative_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi


def compute_dollar_volume(df: pd.DataFrame) -> pd.Series:
    return df["close"] * df["volume"]


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["rvol"] = compute_rvol(df["volume"])
    out["obv"] = compute_obv(df)
    out["cmf"] = compute_cmf(df)
    out["mfi"] = compute_mfi(df)
    out["dollar_volume"] = compute_dollar_volume(df)
    return out
