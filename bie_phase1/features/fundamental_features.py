"""
فیچرهای Fundamental: Fear & Greed, Funding Rate, Open Interest, Long/Short Ratio
"""

import pandas as pd
from db import get_connection


def _load_fear_greed() -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT timestamp, value FROM fear_greed ORDER BY timestamp ASC", conn
        )
    if df.empty:
        return df
    df["timestamp_ms"] = df["timestamp"] * 1000
    return df.sort_values("timestamp_ms")


def _load_funding_rate(symbol: str) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT funding_time, funding_rate FROM funding_rate WHERE symbol = ? ORDER BY funding_time ASC",
            conn, params=(symbol,),
        )
    return df


def _load_open_interest(symbol: str) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT timestamp, open_interest FROM open_interest WHERE symbol = ? ORDER BY timestamp ASC",
            conn, params=(symbol,),
        )
    return df


def compute_fundamental_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    left = pd.DataFrame({"open_time": df["open_time"]}).sort_values("open_time")

    fg = _load_fear_greed()
    if not fg.empty:
        merged = pd.merge_asof(left, fg[["timestamp_ms", "value"]].rename(columns={"timestamp_ms": "open_time"}),
                                on="open_time", direction="backward")
        out["fear_greed"] = merged["value"].values
    else:
        out["fear_greed"] = pd.NA

    fr = _load_funding_rate(symbol)
    if not fr.empty:
        merged = pd.merge_asof(left, fr.rename(columns={"funding_time": "open_time"}),
                                on="open_time", direction="backward")
        out["funding_rate"] = merged["funding_rate"].values
    else:
        out["funding_rate"] = pd.NA

    oi = _load_open_interest(symbol)
    if not oi.empty:
        merged = pd.merge_asof(left, oi.rename(columns={"timestamp": "open_time"}),
                                on="open_time", direction="backward")
        out["open_interest"] = merged["open_interest"].values
    else:
        out["open_interest"] = pd.NA

    out["long_short_ratio"] = pd.NA

    return out
