"""
فاز ۵ - آماده‌سازی دیتاست برای مدل‌های ML/DL
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd

from config import TARGET_HORIZON, TEST_SIZE_RATIO, NON_FEATURE_COLUMNS, MODELS_DIR, USE_SELECTED_FEATURES
from features.build_features import load_feature_store


def load_selected_features(symbol: str, interval: str) -> list:
    """
    اگه models/feature_selection.py قبلاً اجرا شده باشه، لیست فیچرهای منتخب SHAP
    رو می‌خونه؛ وگرنه None. اینجا (نه در feature_selection.py) تعریف شده تا از
    import چرخه‌ای جلوگیری بشه.
    """
    path = os.path.join(MODELS_DIR, f"selected_features_{symbol}_{interval}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def add_target(df: pd.DataFrame, horizon: int = TARGET_HORIZON) -> pd.DataFrame:
    df = df.copy()
    future_close = df["close"].shift(-horizon)
    df["target"] = (future_close > df["close"]).astype("Int64")
    df.loc[future_close.isna(), "target"] = pd.NA
    return df


def select_feature_columns(df: pd.DataFrame) -> list:
    candidate_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    non_empty_cols = [c for c in candidate_cols if df[c].notna().any()]
    return non_empty_cols


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "session" in df.columns:
        session_dummies = pd.get_dummies(df["session"], prefix="session")
        df = pd.concat([df.drop(columns=["session"]), session_dummies], axis=1)

    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    return df


def build_clean_full_dataset(symbol: str, interval: str):
    df = load_feature_store(symbol, interval)
    if df.empty:
        raise ValueError(f"داده‌ای در feature_store برای {symbol}/{interval} نیست. اول pipeline.py رو اجرا کن.")

    df = add_target(df)
    feature_cols = select_feature_columns(df)
    df = clean_dataset(df)

    final_feature_cols = [c for c in feature_cols if c != "session"]
    final_feature_cols += [c for c in df.columns if c.startswith("session_")]
    final_feature_cols = [c for c in final_feature_cols if c in df.columns]

    df = df.dropna(subset=final_feature_cols)
    df = df.sort_values("open_time").reset_index(drop=True)

    if USE_SELECTED_FEATURES:
        selected = load_selected_features(symbol, interval)
        if selected:
            filtered = [c for c in final_feature_cols if c in selected]
            if filtered:
                print(f"[dataset_prep] USE_SELECTED_FEATURES فعاله: از {len(final_feature_cols)} فیچر "
                      f"به {len(filtered)} فیچر منتخب SHAP کاهش یافت.")
                final_feature_cols = filtered
            else:
                print(f"[dataset_prep] هشدار: لیست فیچر منتخب هم‌پوشانی نداره - از کل فیچرها استفاده می‌شه.")
        else:
            print(f"[dataset_prep] USE_SELECTED_FEATURES فعاله ولی هنوز models/feature_selection.py اجرا نشده.")

    return df, final_feature_cols


def build_ml_dataset(symbol: str, interval: str):
    df, final_feature_cols = build_clean_full_dataset(symbol, interval)

    split_idx = int(len(df) * (1 - TEST_SIZE_RATIO))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[final_feature_cols]
    y_train = train_df["target"]
    X_test = test_df[final_feature_cols]
    y_test = test_df["target"]

    return X_train, X_test, y_train, y_test, final_feature_cols, df


if __name__ == "__main__":
    from config import SYMBOL, OHLCV_INTERVAL
    X_train, X_test, y_train, y_test, cols, df = build_ml_dataset(SYMBOL, OHLCV_INTERVAL)
    print(f"تعداد فیچر: {len(cols)}")
    print(f"train: {X_train.shape}, test: {X_test.shape}")
