"""
فیچرهای Trend: EMA, SMA, VWAP, ADX, SuperTrend
"""

import pandas as pd
import numpy as np

from config import EMA_WINDOWS, SMA_WINDOWS, ADX_WINDOW


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def compute_ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_vol_price = (typical_price * df["volume"]).cumsum()
    return cum_vol_price / cum_vol.replace(0, np.nan)


def compute_adx(df: pd.DataFrame, window: int = ADX_WINDOW) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(df)
    atr = tr.ewm(alpha=1 / window, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / window, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / window, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / window, adjust=False).mean()
    return adx


def compute_supertrend(df: pd.DataFrame, window: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    tr = _true_range(df)
    atr = tr.rolling(window=window).mean()

    hl2 = (df["high"] + df["low"]) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    direction = pd.Series(1, index=df.index)
    supertrend = pd.Series(np.nan, index=df.index)

    for i in range(1, len(df)):
        if pd.isna(atr.iloc[i]):
            continue

        if upper_band.iloc[i] < final_upper.iloc[i - 1] or df["close"].iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if lower_band.iloc[i] > final_lower.iloc[i - 1] or df["close"].iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if df["close"].iloc[i] > final_upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < final_lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        supertrend.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    return pd.DataFrame({"supertrend": supertrend, "supertrend_direction": direction})


def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    for w in EMA_WINDOWS:
        out[f"ema_{w}"] = compute_ema(df["close"], w)
    for w in SMA_WINDOWS:
        out[f"sma_{w}"] = compute_sma(df["close"], w)

    out["vwap"] = compute_vwap(df)
    out["adx"] = compute_adx(df)

    st = compute_supertrend(df)
    out["supertrend"] = st["supertrend"]
    out["supertrend_direction"] = st["supertrend_direction"]

    return out
