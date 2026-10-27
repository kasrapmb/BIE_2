"""
فیچرهای Momentum: RSI, MACD, ROC, CCI, Williams %R, Stochastic RSI
"""

import pandas as pd
import numpy as np

from config import RSI_WINDOW, STOCH_RSI_WINDOW


def compute_rsi(series: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_histogram": histogram})


def compute_roc(series: pd.Series, window: int = 12) -> pd.Series:
    return (series - series.shift(window)) / series.shift(window) * 100


def compute_cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    sma = typical_price.rolling(window).mean()
    mean_deviation = typical_price.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (typical_price - sma) / (0.015 * mean_deviation.replace(0, np.nan))
    return cci


def compute_williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    highest_high = df["high"].rolling(window).max()
    lowest_low = df["low"].rolling(window).min()
    wr = (highest_high - df["close"]) / (highest_high - lowest_low).replace(0, np.nan) * -100
    return wr


def compute_stochastic_rsi(series: pd.Series, window: int = STOCH_RSI_WINDOW) -> pd.Series:
    rsi = compute_rsi(series, window)
    min_rsi = rsi.rolling(window).min()
    max_rsi = rsi.rolling(window).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)
    return stoch_rsi


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["rsi"] = compute_rsi(df["close"])

    macd_df = compute_macd(df["close"])
    out["macd"] = macd_df["macd"]
    out["macd_signal"] = macd_df["macd_signal"]
    out["macd_histogram"] = macd_df["macd_histogram"]

    out["roc"] = compute_roc(df["close"])
    out["cci"] = compute_cci(df)
    out["williams_r"] = compute_williams_r(df)
    out["stoch_rsi"] = compute_stochastic_rsi(df["close"])
    return out
