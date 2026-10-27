"""
فاز ۲: Data Validation
هر مشکل احتمالی رو پیدا می‌کنه و در validation_log ثبت می‌کنه، بدون تغییر داده.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
import numpy as np

from config import INTERVAL_MS, OUTLIER_ZSCORE_THRESHOLD
from db import get_connection


def load_raw_ohlcv(symbol: str, interval: str) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query("""
            SELECT symbol, interval, open_time, open, high, low, close, volume, close_time
            FROM ohlcv WHERE symbol = ? AND interval = ? ORDER BY open_time ASC
        """, conn, params=(symbol, interval))
    return df


def _log_issues(rows):
    if not rows:
        return
    detected_at = int(time.time())
    full_rows = [(s, i, ot, issue, detail, detected_at) for (s, i, ot, issue, detail) in rows]
    with get_connection() as conn:
        conn.executemany("""
            INSERT INTO validation_log
            (symbol, interval, open_time, issue_type, detail, detected_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, full_rows)


def check_duplicates(df: pd.DataFrame) -> pd.Series:
    return df["open_time"].duplicated(keep="first")


def check_missing(df: pd.DataFrame) -> pd.Series:
    core_cols = ["open", "high", "low", "close", "volume"]
    return df[core_cols].isna().any(axis=1)


def check_wrong_timestamp(df: pd.DataFrame) -> pd.Series:
    diffs = df["open_time"].diff()
    wrong = diffs <= 0
    wrong.iloc[0] = False
    return wrong


def check_outliers(df: pd.DataFrame, z_threshold: float = OUTLIER_ZSCORE_THRESHOLD) -> pd.Series:
    log_return = np.log(df["close"] / df["close"].shift(1))
    mean = log_return.mean()
    std = log_return.std()
    if std == 0 or pd.isna(std):
        return pd.Series(False, index=df.index)
    z = (log_return - mean) / std
    return z.abs() > z_threshold


def check_invalid_candle(df: pd.DataFrame) -> pd.Series:
    invalid = (
        (df["high"] < df["low"]) |
        (df["high"] < df[["open", "close"]].max(axis=1)) |
        (df["low"] > df[["open", "close"]].min(axis=1))
    )
    return invalid


def check_negative_volume(df: pd.DataFrame) -> pd.Series:
    return df["volume"] < 0


def check_gaps(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    expected_step = INTERVAL_MS.get(interval)
    if expected_step is None:
        raise ValueError(f"interval ناشناخته: {interval}")

    diffs = df["open_time"].diff()
    gap_mask = diffs > expected_step
    gaps = []
    for idx in df.index[gap_mask]:
        prev_time = df.loc[idx - 1, "open_time"] if idx > 0 else None
        curr_time = df.loc[idx, "open_time"]
        if prev_time is None:
            continue
        missing = int((curr_time - prev_time) / expected_step) - 1
        gaps.append({"gap_start": prev_time, "gap_end": curr_time, "missing_candles": missing})
    return pd.DataFrame(gaps)


def validate(symbol: str, interval: str) -> dict:
    df = load_raw_ohlcv(symbol, interval)
    if df.empty:
        print(f"[validation] هیچ داده‌ای برای {symbol} ({interval}) پیدا نشد.")
        return {"total_rows": 0}

    dup_mask = check_duplicates(df)
    missing_mask = check_missing(df)
    wrong_ts_mask = check_wrong_timestamp(df)
    outlier_mask = check_outliers(df)
    invalid_mask = check_invalid_candle(df)
    neg_vol_mask = check_negative_volume(df)
    gaps_df = check_gaps(df, interval)

    log_rows = []
    for mask, issue_type in [
        (dup_mask, "duplicate"), (missing_mask, "missing"), (wrong_ts_mask, "wrong_timestamp"),
        (outlier_mask, "outlier"), (invalid_mask, "invalid_candle"), (neg_vol_mask, "negative_volume"),
    ]:
        for idx in df.index[mask]:
            log_rows.append((symbol, interval, int(df.loc[idx, "open_time"]), issue_type, None))

    for _, g in gaps_df.iterrows():
        detail = f"missing_candles={int(g['missing_candles'])}, gap_end={int(g['gap_end'])}"
        log_rows.append((symbol, interval, int(g["gap_start"]), "gap", detail))

    _log_issues(log_rows)

    report = {
        "total_rows": len(df),
        "duplicates": int(dup_mask.sum()),
        "missing": int(missing_mask.sum()),
        "wrong_timestamp": int(wrong_ts_mask.sum()),
        "outliers": int(outlier_mask.sum()),
        "invalid_candles": int(invalid_mask.sum()),
        "negative_volume": int(neg_vol_mask.sum()),
        "gaps_found": len(gaps_df),
        "total_missing_candles": int(gaps_df["missing_candles"].sum()) if not gaps_df.empty else 0,
    }

    print(f"[validation] {symbol} ({interval}) → {report}")
    return report


def run(symbol: str, interval: str):
    return validate(symbol, interval)


if __name__ == "__main__":
    from config import SYMBOL, OHLCV_INTERVAL
    run(SYMBOL, OHLCV_INTERVAL)
