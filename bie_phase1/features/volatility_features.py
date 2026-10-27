"""
فیچرهای Volatility: ATR, Historical Volatility, Parkinson, Rolling Std
"""

import pandas as pd
import numpy as np


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def compute_historical_volatility(df: pd.DataFrame, window: int = 20, annualize_periods: int = None) -> pd.Series:
    log_return = np.log(df["close"] / df["close"].shift(1))
    vol = log_return.rolling(window).std()
    if annualize_periods:
        vol = vol * np.sqrt(annualize_periods)
    return vol


def compute_parkinson_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    log_hl = np.log(df["high"] / df["low"])
    factor = 1 / (4 * np.log(2))
    parkinson_var = factor * (log_hl ** 2)
    return np.sqrt(parkinson_var.rolling(window).mean())


def compute_rolling_std(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return df["close"].rolling(window).std()


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["atr"] = compute_atr(df)
    out["historical_volatility"] = compute_historical_volatility(df)
    out["parkinson_volatility"] = compute_parkinson_volatility(df)
    out["rolling_std"] = compute_rolling_std(df)
    return out
