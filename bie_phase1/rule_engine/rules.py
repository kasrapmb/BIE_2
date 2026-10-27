"""
فاز ۶ - Rule Engine: قوانین قطعی بازار (EMA Cross, RSI, Breakout, Liquidity Sweep, CHOCH)
هر قانون عددی در [-1, +1] برمی‌گردونه.
"""

import pandas as pd
import numpy as np

RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30


def rule_ema_cross(df: pd.DataFrame) -> pd.Series:
    fast, slow = df["ema_9"], df["ema_21"]
    cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    signal = pd.Series(0.0, index=df.index)
    signal[cross_up] = 1.0
    signal[cross_down] = -1.0
    return signal


def rule_rsi(df: pd.DataFrame) -> pd.Series:
    rsi = df["rsi"]
    signal = pd.Series(0.0, index=df.index)
    signal[rsi < RSI_OVERSOLD] = 0.6
    signal[rsi > RSI_OVERBOUGHT] = -0.6
    return signal


def rule_breakout(df: pd.DataFrame) -> pd.Series:
    if "bos" not in df.columns:
        return pd.Series(0.0, index=df.index)
    return df["bos"].astype(float)


def rule_liquidity_sweep(df: pd.DataFrame) -> pd.Series:
    signal = pd.Series(0.0, index=df.index)
    if "liquidity_sweep_high" in df.columns:
        signal[df["liquidity_sweep_high"] == 1] -= 0.7
    if "liquidity_sweep_low" in df.columns:
        signal[df["liquidity_sweep_low"] == 1] += 0.7
    return signal


def rule_choch(df: pd.DataFrame) -> pd.Series:
    if "choch" not in df.columns or "bos" not in df.columns:
        return pd.Series(0.0, index=df.index)
    signal = pd.Series(0.0, index=df.index)
    is_choch = df["choch"] == 1
    signal[is_choch] = df.loc[is_choch, "bos"].astype(float) * 0.8
    return signal


RULES = {
    "ema_cross": rule_ema_cross,
    "rsi": rule_rsi,
    "breakout": rule_breakout,
    "liquidity_sweep": rule_liquidity_sweep,
    "choch": rule_choch,
}


def compute_rule_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for name, func in RULES.items():
        out[f"rule_{name}"] = func(df)

    rule_cols = [f"rule_{name}" for name in RULES.keys()]
    out["rule_engine_score"] = out[rule_cols].mean(axis=1)
    return out


if __name__ == "__main__":
    from config import SYMBOL, OHLCV_INTERVAL
    from features.build_features import load_feature_store
    df = load_feature_store(SYMBOL, OHLCV_INTERVAL)
    if df.empty:
        print("داده‌ای نیست - اول pipeline.py رو اجرا کن.")
    else:
        signals = compute_rule_signals(df)
        print(signals.tail(10).to_string())
