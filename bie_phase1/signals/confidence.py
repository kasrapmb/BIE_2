"""
فاز ۸ - Confidence Engine: Agreement + Historical Accuracy + Risk + Market Regime
"""

import pandas as pd
import numpy as np


def compute_model_agreement(proba_df: pd.DataFrame) -> pd.Series:
    proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
    if len(proba_cols) < 2:
        return pd.Series(0.5, index=proba_df.index)
    std = proba_df[proba_cols].std(axis=1)
    agreement = 1 - (std / 0.5).clip(upper=1)
    return agreement


def compute_historical_accuracy(df: pd.DataFrame, final_signal: pd.Series, target: pd.Series, lookback: int = 50) -> pd.Series:
    is_correct = pd.Series(np.nan, index=df.index)
    is_buy = final_signal == "BUY"
    is_sell = final_signal == "SELL"
    is_correct[is_buy] = (target[is_buy] == 1).astype(float)
    is_correct[is_sell] = (target[is_sell] == 0).astype(float)

    rolling_accuracy = is_correct.rolling(lookback, min_periods=5).mean()
    return rolling_accuracy.fillna(0.5)


def compute_risk_penalty(df: pd.DataFrame) -> pd.Series:
    if "atr" not in df.columns or "close" not in df.columns:
        return pd.Series(1.0, index=df.index)
    atr_pct = df["atr"] / df["close"]
    rolling_median_atr_pct = atr_pct.rolling(100, min_periods=10).median()
    relative_risk = (atr_pct / rolling_median_atr_pct.replace(0, np.nan)).clip(lower=0.5, upper=3)
    penalty = 1 / relative_risk
    return penalty.fillna(1.0).clip(upper=1.0)


def compute_market_regime(df: pd.DataFrame) -> pd.Series:
    if "adx" not in df.columns:
        return pd.Series(0.7, index=df.index)
    return (df["adx"] / 50).clip(upper=1.0).fillna(0.5)


def compute_confidence(df: pd.DataFrame, proba_df: pd.DataFrame, final_signal: pd.Series, target: pd.Series) -> pd.DataFrame:
    agreement = compute_model_agreement(proba_df)
    hist_accuracy = compute_historical_accuracy(df, final_signal, target)
    risk_penalty = compute_risk_penalty(df)
    regime_score = compute_market_regime(df)

    common_idx = proba_df.index
    confidence = pd.DataFrame({
        "agreement": agreement,
        "historical_accuracy": hist_accuracy.loc[common_idx],
        "risk_penalty": risk_penalty.loc[common_idx],
        "market_regime": regime_score.loc[common_idx],
    })
    confidence["confidence"] = confidence.mean(axis=1)

    return confidence
