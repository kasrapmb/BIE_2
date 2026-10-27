"""
Feature Selection با SHAP - از فیچرها، مهم‌ترین‌ها رو (پیش‌فرض ۲۵ تا) شناسایی می‌کنه.
روی داده‌ی train حساب می‌شه (نه test)، تا خودِ انتخاب فیچر به تست نشتی پیدا نکنه.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier

from config import MODELS_DIR, FEATURE_SELECTION_TOP_N, BASE_DIR
from db import init_db
from models.dataset_prep import build_ml_dataset, load_selected_features


def compute_shap_importance(symbol: str, interval: str) -> pd.DataFrame:
    init_db()
    X_train, X_test, y_train, y_test, feature_cols, _ = build_ml_dataset(symbol, interval)

    model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                                random_state=42, verbose=False)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)

    result = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap_direction": mean_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return result


def select_top_features(symbol: str, interval: str, top_n: int = FEATURE_SELECTION_TOP_N) -> list:
    importance_df = compute_shap_importance(symbol, interval)
    top_features = importance_df.head(top_n)["feature"].tolist()

    os.makedirs(MODELS_DIR, exist_ok=True)
    selected_path = os.path.join(MODELS_DIR, f"selected_features_{symbol}_{interval}.json")
    with open(selected_path, "w", encoding="utf-8") as f:
        json.dump(top_features, f, ensure_ascii=False, indent=2)

    _save_full_report(importance_df, symbol, interval, top_n)

    print(f"[feature_selection] {len(top_features)} فیچر برتر (از {len(importance_df)} فیچر) انتخاب و ذخیره شد:")
    print(f"[feature_selection] {selected_path}")

    return top_features


def _save_full_report(importance_df: pd.DataFrame, symbol: str, interval: str, top_n: int, base_dir: str = BASE_DIR):
    json_path = os.path.join(base_dir, "feature_importance_shap.json")
    importance_df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    md_lines = [
        f"# اهمیت فیچرها (SHAP) — {symbol} ({interval})",
        "",
        f"روی داده‌ی train محاسبه شده. خط قرمز: {top_n} فیچر برتر انتخاب شدن.",
        "",
        "| رتبه | فیچر | میانگین |SHAP| | جهت تأثیر |",
        "|---|---|---|---|",
    ]
    for i, row in importance_df.iterrows():
        direction = "➕ صعودی" if row["mean_shap_direction"] > 0 else "➖ نزولی"
        marker = " ✅" if i < top_n else ""
        md_lines.append(f"| {i+1}{marker} | {row['feature']} | {row['mean_abs_shap']:.5f} | {direction} |")

    md_path = os.path.join(base_dir, "feature_importance_shap.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[feature_selection] گزارش کامل: {md_path}")


def run(symbol: str, interval: str, top_n: int = FEATURE_SELECTION_TOP_N):
    return select_top_features(symbol, interval, top_n)


if __name__ == "__main__":
    import argparse
    from config import SYMBOL, OHLCV_INTERVAL
    parser = argparse.ArgumentParser(description="Feature Selection با SHAP")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--interval", default=OHLCV_INTERVAL)
    parser.add_argument("--top-n", type=int, default=FEATURE_SELECTION_TOP_N)
    args = parser.parse_args()
    run(args.symbol, args.interval, args.top_n)
