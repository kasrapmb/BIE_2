"""
فاز ۷ - Signal Aggregator: ترکیب ML + Rule Engine به یک امتیاز/سیگنال نهایی
"""

import pandas as pd
import numpy as np

from config import SIGNAL_WEIGHTS, BUY_THRESHOLD, SELL_THRESHOLD


def aggregate_ml_scores(proba_df: pd.DataFrame) -> pd.Series:
    proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
    if not proba_cols:
        return pd.Series(0.5, index=proba_df.index)
    return proba_df[proba_cols].mean(axis=1)


def rule_score_to_probability(rule_score: pd.Series) -> pd.Series:
    return (rule_score + 1) / 2


def aggregate_signals(ml_proba: pd.Series, rule_score: pd.Series, weights: dict = SIGNAL_WEIGHTS) -> pd.DataFrame:
    rule_proba = rule_score_to_probability(rule_score)

    final_score = (
        weights.get("ml", 0.5) * ml_proba +
        weights.get("rule_engine", 0.5) * rule_proba
    )

    signal = pd.Series("HOLD", index=final_score.index)
    signal[final_score >= BUY_THRESHOLD] = "BUY"
    signal[final_score <= SELL_THRESHOLD] = "SELL"

    return pd.DataFrame({
        "ml_score": ml_proba, "rule_score": rule_proba,
        "final_score": final_score, "final_signal": signal,
    })
