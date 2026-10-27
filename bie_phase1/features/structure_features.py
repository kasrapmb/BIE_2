"""
فیچرهای Market Structure: Support, Resistance, BOS, CHOCH, Liquidity Sweep,
Equal High, Equal Low, Swing High, Swing Low

⚠️ درباره‌ی نشتی اطلاعات (leakage): تشخیص "این کندل swing high بوده" ذاتاً نیاز
به دیدن `lookback` کندل *بعدی* داره (باید مطمئن بشیم قیمت بعداً بالاتر نرفته).
یعنی در لحظه‌ی وقوع کندل i، امکان ندارد بفهمی swing point بوده - فقط lookback
کندل بعدتر می‌فهمی. برای همین همه‌ی سیگنال‌های این فایل (boolean flag ها *و*
مقدار قیمت مرتبط باهاشون) با shift(lookback) به تأخیر افتادن - دقیقاً از همون
لحظه‌ای که واقعاً "قابل‌دونستن" بودن وارد فیچرها می‌شن، نه زودتر.
"""

import pandas as pd
import numpy as np

from config import SWING_LOOKBACK, EQUAL_HL_TOLERANCE


def compute_swing_points(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> pd.DataFrame:
    """
    خروجی: swing_high, swing_low (boolean ۰/۱) + swing_high_price, swing_low_price
    (قیمت واقعی swing، هم‌زمان با پرچم به تأخیر افتاده).
    """
    high, low = df["high"], df["low"]

    is_swing_high_raw = pd.Series(False, index=df.index)
    is_swing_low_raw = pd.Series(False, index=df.index)

    for i in range(lookback, len(df) - lookback):
        window_high = high.iloc[i - lookback: i + lookback + 1]
        window_low = low.iloc[i - lookback: i + lookback + 1]
        if high.iloc[i] == window_high.max() and (window_high == window_high.max()).sum() == 1:
            is_swing_high_raw.iloc[i] = True
        if low.iloc[i] == window_low.min() and (window_low == window_low.min()).sum() == 1:
            is_swing_low_raw.iloc[i] = True

    swing_high_price_raw = high.where(is_swing_high_raw)
    swing_low_price_raw = low.where(is_swing_low_raw)

    is_swing_high = is_swing_high_raw.shift(lookback).fillna(False)
    is_swing_low = is_swing_low_raw.shift(lookback).fillna(False)
    swing_high_price = swing_high_price_raw.shift(lookback)
    swing_low_price = swing_low_price_raw.shift(lookback)

    out = pd.DataFrame(index=df.index)
    out["swing_high"] = is_swing_high.astype(int)
    out["swing_low"] = is_swing_low.astype(int)
    out["swing_high_price"] = swing_high_price
    out["swing_low_price"] = swing_low_price
    return out


def compute_support_resistance(df: pd.DataFrame, swings: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["resistance"] = swings["swing_high_price"].ffill()
    out["support"] = swings["swing_low_price"].ffill()
    return out


def compute_bos_choch(df: pd.DataFrame, swings: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    last_swing_high = swings["swing_high_price"].ffill()
    last_swing_low = swings["swing_low_price"].ffill()

    bos_bullish = df["close"] > last_swing_high
    bos_bearish = df["close"] < last_swing_low

    bos_direction = pd.Series(0, index=df.index)
    bos_direction[bos_bullish] = 1
    bos_direction[bos_bearish] = -1

    out["bos"] = bos_direction
    last_nonzero = bos_direction.replace(0, np.nan).ffill().shift(1)
    out["choch"] = ((bos_direction != 0) & (last_nonzero != 0) & (bos_direction != last_nonzero)).astype(int)

    return out


def compute_equal_high_low(df: pd.DataFrame, swings: pd.DataFrame, tolerance: float = EQUAL_HL_TOLERANCE) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    last_swing_high = swings["swing_high_price"].ffill()
    last_swing_low = swings["swing_low_price"].ffill()

    out["equal_high"] = ((df["high"] - last_swing_high).abs() / last_swing_high < tolerance).astype(int)
    out["equal_low"] = ((df["low"] - last_swing_low).abs() / last_swing_low < tolerance).astype(int)
    return out


def compute_liquidity_sweep(df: pd.DataFrame, swings: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    last_swing_high = swings["swing_high_price"].ffill()
    last_swing_low = swings["swing_low_price"].ffill()

    sweep_high = (df["high"] > last_swing_high) & (df["close"] < last_swing_high)
    sweep_low = (df["low"] < last_swing_low) & (df["close"] > last_swing_low)

    out["liquidity_sweep_high"] = sweep_high.astype(int)
    out["liquidity_sweep_low"] = sweep_low.astype(int)
    return out


def compute_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    swings = compute_swing_points(df)
    sr = compute_support_resistance(df, swings)
    bos_choch = compute_bos_choch(df, swings)
    equal_hl = compute_equal_high_low(df, swings)
    sweep = compute_liquidity_sweep(df, swings)

    swings_public = swings[["swing_high", "swing_low"]]

    return pd.concat([swings_public, sr, bos_choch, equal_hl, sweep], axis=1)
