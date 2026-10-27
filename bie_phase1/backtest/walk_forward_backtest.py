"""
Walk-Forward Backtest: ترکیب Walk-Forward Validation با Backtest واقعی
هر fold با مدلی که فقط گذشته رو دیده train می‌شه، سیگنال روی فصل بعد تولید می‌شه،
و همه‌ی این فصل‌های تست پشت‌سرهم به یک equity curve پیوسته وصل می‌شن.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import WALK_FORWARD_FREQ, WALK_FORWARD_MIN_TRAIN_PERIODS
from db import init_db
from models.dataset_prep import build_clean_full_dataset
from models.train_ml import _build_models
from models.walk_forward import generate_folds
from rule_engine.rules import compute_rule_signals
from signals.aggregator import aggregate_ml_scores, aggregate_signals
from backtest.engine import run_backtest, run_backtest_advanced


def run_walk_forward_backtest(symbol: str, interval: str,
                               freq: str = WALK_FORWARD_FREQ,
                               min_train_periods: int = WALK_FORWARD_MIN_TRAIN_PERIODS,
                               use_advanced_engine: bool = True) -> dict:
    init_db()
    df, feature_cols = build_clean_full_dataset(symbol, interval)

    folds = generate_folds(df, freq=freq, min_train_periods=min_train_periods)
    if not folds:
        raise SystemExit(
            f"\n[walk_forward_backtest] هیچ fold معتبری ساخته نشد - دیتای تاریخی کافی نیست.\n"
            f"'--min-train-periods' رو کم کن یا '--freq M' رو امتحان کن.\n"
        )

    print(f"[walk_forward_backtest] {len(folds)} fold برای backtest پیوسته ساخته شد.\n")

    all_signals = []
    all_ml_scores = []
    all_index = []

    for fold in folds:
        train_df = df.loc[fold["train_index"]]
        test_df = df.loc[fold["test_index"]]

        X_train, y_train = train_df[feature_cols], train_df["target"]
        X_test = test_df[feature_cols]

        models = _build_models()
        proba_df = pd.DataFrame(index=X_test.index)
        for name, model in models.items():
            model.fit(X_train, y_train)
            proba_df[f"proba_{name}"] = model.predict_proba(X_test)[:, 1]

        ml_score = aggregate_ml_scores(proba_df)
        rule_signals_df = compute_rule_signals(df.loc[fold["test_index"]])
        rule_score = rule_signals_df["rule_engine_score"]

        agg = aggregate_signals(ml_score, rule_score)

        all_signals.append(agg["final_signal"])
        all_ml_scores.append(ml_score)
        all_index.extend(fold["test_index"].tolist())

        print(f"[walk_forward_backtest] Fold {fold['fold']} ({fold['test_period']}): "
              f"{len(X_test)} کندل سیگنال گرفت.")

    combined_signal = pd.concat(all_signals).sort_index()
    combined_proba = pd.concat(all_ml_scores).sort_index()
    combined_df = df.loc[combined_signal.index].sort_index()

    print(f"\n[walk_forward_backtest] مجموع {len(combined_signal)} کندل خارج از نمونه (out-of-sample) "
          f"برای backtest پیوسته آماده شد.\n")

    if use_advanced_engine and "atr" in combined_df.columns:
        result = run_backtest_advanced(combined_df, combined_signal, prediction_proba=combined_proba)
    else:
        if use_advanced_engine:
            print("[walk_forward_backtest] هشدار: ستون 'atr' پیدا نشد - از موتور ساده استفاده می‌شه.")
        result = run_backtest(combined_df, combined_signal)

    result["n_folds"] = len(folds)
    result["symbol"] = symbol
    result["interval"] = interval
    result["freq"] = freq
    return result


def run(symbol: str, interval: str, **kwargs):
    return run_walk_forward_backtest(symbol, interval, **kwargs)


if __name__ == "__main__":
    from config import SYMBOL, OHLCV_INTERVAL
    result = run(SYMBOL, OHLCV_INTERVAL)
    print(result["metrics"])
