"""
فاز ۳: Cleaning
داده خام رو تمیز و resample می‌کنه؛ normalize عمداً اینجا انجام نمی‌شه.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
import numpy as np

from config import INTERVAL_MS, MAX_CONSECUTIVE_FILL
from db import get_connection
from validation.validators import load_raw_ohlcv


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["open_time"], keep="last")


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("open_time").reset_index(drop=True)


def resample_to_grid(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    step = INTERVAL_MS.get(interval)
    if step is None:
        raise ValueError(f"interval ناشناخته: {interval}")

    if df.empty:
        return df

    full_grid = np.arange(df["open_time"].min(), df["open_time"].max() + step, step)
    df_indexed = df.set_index("open_time")
    df_reindexed = df_indexed.reindex(full_grid)
    df_reindexed.index.name = "open_time"
    df_reindexed = df_reindexed.reset_index()
    df_reindexed["symbol"] = df["symbol"].iloc[0]
    df_reindexed["interval"] = interval
    return df_reindexed


def fill_missing(df: pd.DataFrame, max_fill: int = MAX_CONSECUTIVE_FILL) -> pd.DataFrame:
    df = df.copy()
    df["is_imputed"] = df["close"].isna().astype(int)

    df["close"] = df["close"].ffill(limit=max_fill)
    df["open"] = df["open"].fillna(df["close"])
    df["high"] = df["high"].fillna(df["close"])
    df["low"] = df["low"].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)

    if "close_time" in df.columns:
        step_guess = int(df["open_time"].diff().median())
        df["close_time"] = df["close_time"].fillna(df["open_time"] + step_guess - 1)

    before = len(df)
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"[cleaning] {dropped} کندل به‌خاطر گپ بزرگ‌تر از حد مجاز حذف شد.")

    return df


def save_clean(df: pd.DataFrame):
    if df.empty:
        print("[cleaning] چیزی برای ذخیره وجود نداره.")
        return
    cleaned_at = int(time.time())
    rows = list(df.apply(
        lambda r: (
            r["symbol"], r["interval"], int(r["open_time"]),
            float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]),
            float(r["volume"]), int(r["close_time"]), int(r["is_imputed"]), cleaned_at,
        ), axis=1,
    ))
    with get_connection() as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO ohlcv_clean
            (symbol, interval, open_time, open, high, low, close, volume, close_time, is_imputed, cleaned_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
    print(f"[cleaning] {len(rows)} کندل تمیزشده ذخیره شد.")


def load_clean_ohlcv(symbol: str, interval: str) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query("""
            SELECT symbol, interval, open_time, open, high, low, close, volume, close_time, is_imputed
            FROM ohlcv_clean WHERE symbol = ? AND interval = ? ORDER BY open_time ASC
        """, conn, params=(symbol, interval))
    return df


def clean(symbol: str, interval: str) -> pd.DataFrame:
    df = load_raw_ohlcv(symbol, interval)
    if df.empty:
        print(f"[cleaning] هیچ داده خامی برای {symbol} ({interval}) پیدا نشد.")
        return df

    df = remove_duplicates(df)
    df = sort_by_time(df)
    df = resample_to_grid(df, interval)
    df = fill_missing(df)
    save_clean(df)
    return df


def run(symbol: str, interval: str):
    return clean(symbol, interval)


if __name__ == "__main__":
    from config import SYMBOL, OHLCV_INTERVAL
    run(SYMBOL, OHLCV_INTERVAL)
