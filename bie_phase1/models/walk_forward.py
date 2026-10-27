"""
Walk-Forward Validation (Expanding Window)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from config import WALK_FORWARD_FREQ, WALK_FORWARD_MIN_TRAIN_PERIODS, MIN_ROWS_FOR_TRAINING
from db import init_db
from models.dataset_prep import build_clean_full_dataset
from models.train_ml import _build_models
from rule_engine.rules import compute_rule_signals
from features.build_features import load_feature_store


def _to_periods(df: pd.DataFrame, freq: str) -> pd.Series:
    dt = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    return dt.dt.to_period(freq)


def generate_folds(df: pd.DataFrame, freq: str = WALK_FORWARD_FREQ,
                    min_train_periods: int = WALK_FORWARD_MIN_TRAIN_PERIODS) -> list:
    periods = _to_periods(df, freq)
    unique_periods = sorted(periods.unique())

    folds = []
    for i in range(min_train_periods, len(unique_periods)):
        test_period = unique_periods[i]
        train_mask = periods < test_period
        test_mask = periods == test_period

        train_idx = df.index[train_mask]
        test_idx = df.index[test_mask]

        if len(train_idx) < MIN_ROWS_FOR_TRAINING or len(test_idx) == 0:
            continue

        folds.append({
            "fold": len(folds) + 1,
            "test_period": str(test_period),
            "train_index": train_idx,
            "test_index": test_idx,
        })

    return folds


def _evaluate_fold(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "true_negative": int(cm[0][0]), "false_positive": int(cm[0][1]),
        "false_negative": int(cm[1][0]), "true_positive": int(cm[1][1]),
    }


def _evaluate_rule_engine_fold(df: pd.DataFrame, test_idx) -> dict:
    rule_signals = compute_rule_signals(df)
    rule_score = rule_signals.loc[test_idx, "rule_engine_score"]
    directional = rule_score[rule_score != 0]
    if directional.empty:
        return {"signals_fired": 0, "win_rate": None}
    predicted_up = directional > 0
    actual_up = df.loc[directional.index, "target"] == 1
    win_rate = (predicted_up == actual_up).mean()
    return {"signals_fired": int(len(directional)), "win_rate": round(float(win_rate), 4)}


def run_walk_forward(symbol: str, interval: str,
                      freq: str = WALK_FORWARD_FREQ,
                      min_train_periods: int = WALK_FORWARD_MIN_TRAIN_PERIODS) -> dict:
    init_db()
    df, feature_cols = build_clean_full_dataset(symbol, interval)

    folds = generate_folds(df, freq=freq, min_train_periods=min_train_periods)
    if not folds:
        raise SystemExit(
            f"\n[walk_forward] هیچ fold معتبری ساخته نشد. یا داده کافی نیست، یا "
            f"WALK_FORWARD_MIN_TRAIN_PERIODS (={min_train_periods}) نسبت به طول تاریخچه‌ات زیاده.\n"
            f"یا freq رو عوض کن (مثلاً 'M' به‌جای 'Q')، یا داده تاریخی بیشتر جمع کن.\n"
        )

    print(f"[walk_forward] {len(folds)} fold ساخته شد (freq={freq}, min_train_periods={min_train_periods})\n")

    fold_results = []
    rule_fold_results = []

    for fold in folds:
        train_df = df.loc[fold["train_index"]]
        test_df = df.loc[fold["test_index"]]

        X_train, y_train = train_df[feature_cols], train_df["target"]
        X_test, y_test = test_df[feature_cols], test_df["target"]

        print(f"[walk_forward] Fold {fold['fold']}: train={len(X_train)} ردیف, "
              f"test={len(X_test)} ردیف (دوره تست: {fold['test_period']})")

        models = _build_models()
        fold_row = {"fold": fold["fold"], "test_period": fold["test_period"],
                    "train_rows": len(X_train), "test_rows": len(X_test)}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = _evaluate_fold(y_test, y_pred)
            fold_row[name] = metrics

        fold_results.append(fold_row)
        rule_fold_results.append({
            "fold": fold["fold"], "test_period": fold["test_period"],
            **_evaluate_rule_engine_fold(df, fold["test_index"]),
        })

    model_names = [n for n in fold_results[0].keys() if n not in ("fold", "test_period", "train_rows", "test_rows")]
    summary = {}
    for name in model_names:
        for metric in ["accuracy", "precision", "recall", "f1"]:
            values = [f[name][metric] for f in fold_results]
            summary.setdefault(name, {})[f"mean_{metric}"] = round(float(np.mean(values)), 4)
            summary.setdefault(name, {})[f"std_{metric}"] = round(float(np.std(values)), 4)

    valid_win_rates = [r["win_rate"] for r in rule_fold_results if r["win_rate"] is not None]
    rule_engine_summary = {
        "folds_with_signal": len(valid_win_rates),
        "mean_win_rate": round(float(np.mean(valid_win_rates)), 4) if valid_win_rates else None,
        "std_win_rate": round(float(np.std(valid_win_rates)), 4) if valid_win_rates else None,
    }

    return {
        "symbol": symbol, "interval": interval, "freq": freq, "n_folds": len(folds),
        "folds": fold_results, "summary": summary,
        "rule_engine_folds": rule_fold_results, "rule_engine_summary": rule_engine_summary,
    }


def run(symbol: str, interval: str, **kwargs):
    return run_walk_forward(symbol, interval, **kwargs)


if __name__ == "__main__":
    import json
    from config import SYMBOL, OHLCV_INTERVAL
    result = run(SYMBOL, OHLCV_INTERVAL)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
