"""
فاز ۵ (تکمیلی) - Predict: لود مدل‌های آموزش‌دیده و پیش‌بینی روی داده جدید
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
import pandas as pd

from config import MODELS_DIR

MODEL_NAMES = ["random_forest", "lightgbm", "xgboost", "catboost"]


def load_models(symbol: str, interval: str) -> dict:
    models = {}
    for name in MODEL_NAMES:
        path = os.path.join(MODELS_DIR, f"{name}_{symbol}_{interval}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


def load_feature_columns(symbol: str, interval: str) -> list:
    path = os.path.join(MODELS_DIR, f"feature_columns_{symbol}_{interval}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"لیست فیچرها پیدا نشد: {path} - اول train_ml.py رو اجرا کن.")
    with open(path) as f:
        return json.load(f)


def predict_proba_up(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    for name, model in models.items():
        proba = model.predict_proba(X)[:, 1]
        out[f"proba_{name}"] = proba
    return out


if __name__ == "__main__":
    from config import SYMBOL, OHLCV_INTERVAL
    from models.dataset_prep import build_ml_dataset

    models = load_models(SYMBOL, OHLCV_INTERVAL)
    if not models:
        print("هیچ مدل آموزش‌دیده‌ای پیدا نشد - اول models/train_ml.py رو اجرا کن.")
    else:
        _, X_test, _, y_test, _, _ = build_ml_dataset(SYMBOL, OHLCV_INTERVAL)
        proba_df = predict_proba_up(models, X_test)
        print(proba_df.tail(10).to_string())
