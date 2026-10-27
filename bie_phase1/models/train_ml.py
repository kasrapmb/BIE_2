"""
فاز ۵ - ML Layer: آموزش LightGBM, XGBoost, CatBoost, Random Forest
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from config import MODELS_DIR
from db import init_db
from models.dataset_prep import build_ml_dataset


def _build_models() -> dict:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_child_samples=10, random_state=42, verbosity=-1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42, eval_metric="logloss", verbosity=0,
        ),
        "catboost": CatBoostClassifier(
            iterations=200, depth=5, learning_rate=0.05, random_state=42, verbose=False,
        ),
    }


def _evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "test_samples": int(len(y_test)),
    }


def train_all_models(symbol: str, interval: str) -> dict:
    init_db()
    X_train, X_test, y_train, y_test, feature_cols, _ = build_ml_dataset(symbol, interval)

    if len(X_train) < 30:
        raise ValueError(f"داده کافی برای آموزش نیست (فقط {len(X_train)} ردیف train).")

    os.makedirs(MODELS_DIR, exist_ok=True)
    models = _build_models()
    results = {}

    for name, model in models.items():
        print(f"[train_ml] در حال آموزش {name} ...")
        model.fit(X_train, y_train)
        metrics = _evaluate(model, X_test, y_test)
        results[name] = metrics

        model_path = os.path.join(MODELS_DIR, f"{name}_{symbol}_{interval}.joblib")
        joblib.dump(model, model_path)
        print(f"[train_ml] {name} ذخیره شد → accuracy={metrics['accuracy']}, f1={metrics['f1']}")

    feature_cols_path = os.path.join(MODELS_DIR, f"feature_columns_{symbol}_{interval}.json")
    with open(feature_cols_path, "w") as f:
        json.dump(feature_cols, f)

    majority_class_ratio = max(y_test.mean(), 1 - y_test.mean())
    results["baseline_majority_class"] = {
        "accuracy": round(float(majority_class_ratio), 4),
        "note": "اگه مدل بهتر از این نباشه، یعنی چیزی یاد نگرفته",
    }

    return results


def run(symbol: str, interval: str):
    return train_all_models(symbol, interval)


if __name__ == "__main__":
    from config import SYMBOL, OHLCV_INTERVAL
    results = run(SYMBOL, OHLCV_INTERVAL)
    print(json.dumps(results, indent=2, ensure_ascii=False))
