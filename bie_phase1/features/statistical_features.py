"""
فیچرهای Statistical: Mean, Median, Variance, Skewness, Kurtosis, Entropy, Z-score
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy

from config import ROLLING_STATS_WINDOW


def compute_statistical_features(df: pd.DataFrame, window: int = ROLLING_STATS_WINDOW) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    log_return = np.log(df["close"] / df["close"].shift(1))

    out["stat_mean"] = log_return.rolling(window).mean()
    out["stat_median"] = log_return.rolling(window).median()
    out["stat_variance"] = log_return.rolling(window).var()
    out["stat_skewness"] = log_return.rolling(window).apply(lambda x: skew(x, nan_policy="omit"), raw=True)
    out["stat_kurtosis"] = log_return.rolling(window).apply(lambda x: kurtosis(x, nan_policy="omit"), raw=True)

    def _rolling_entropy(x):
        if np.all(np.isnan(x)):
            return np.nan
        hist, _ = np.histogram(x[~np.isnan(x)], bins=10)
        if hist.sum() == 0:
            return np.nan
        return entropy(hist + 1e-9)

    out["stat_entropy"] = log_return.rolling(window).apply(_rolling_entropy, raw=True)

    rolling_mean = df["close"].rolling(window).mean()
    rolling_std = df["close"].rolling(window).std()
    out["stat_zscore"] = (df["close"] - rolling_mean) / rolling_std.replace(0, np.nan)

    return out
