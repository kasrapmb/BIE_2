"""
فاز ۹ - Explainability: مهم‌ترین فیچرها + کدوم قانون فعال بوده
"""

import pandas as pd
import numpy as np


def get_global_feature_importance(models: dict, feature_columns: list, top_n: int = 15) -> pd.DataFrame:
    importances = pd.DataFrame(index=feature_columns)
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            importances[name] = model.feature_importances_
    if importances.empty:
        return pd.DataFrame(columns=["feature", "importance"])

    importances["avg_importance"] = importances.mean(axis=1)
    result = importances[["avg_importance"]].sort_values("avg_importance", ascending=False).head(top_n)
    result = result.reset_index().rename(columns={"index": "feature", "avg_importance": "importance"})
    return result


def explain_single_signal(row_features: pd.Series, global_importance: pd.DataFrame,
                           rule_signals_row: pd.Series, top_n: int = 5) -> dict:
    top_features = global_importance.head(top_n)["feature"].tolist()
    feature_values = {f: (None if f not in row_features or pd.isna(row_features[f]) else float(row_features[f]))
                       for f in top_features}

    active_rules = {
        col: float(val) for col, val in rule_signals_row.items()
        if col.startswith("rule_") and col != "rule_engine_score" and val != 0
    }

    return {"top_contributing_features": feature_values, "active_rules": active_rules}
