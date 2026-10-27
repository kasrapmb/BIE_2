"""
فیچرهای Price: Return, Log Return, Gap, Momentum, Acceleration
"""

import pandas as pd
import numpy as np


def compute_price_features(df: pd.DataFrame, momentum_window: int = 10) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["return"] = df["close"].pct_change()
    out["log_return"] = np.log(df["close"] / df["close"].shift(1))
    out["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    out["momentum"] = df["close"] - df["close"].shift(momentum_window)
    out["acceleration"] = out["momentum"] - out["momentum"].shift(momentum_window)
    return out
