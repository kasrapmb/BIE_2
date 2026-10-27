"""
گزارش Accuracy - فایل مستقل برای دیدن دقت واقعی موتور
اجرا: python accuracy_report.py [--no-retrain]
"""

import os
import json
from sklearn.metrics import confusion_matrix

from config import SYMBOL, OHLCV_INTERVAL, BASE_DIR
from db import init_db, get_connection
from models.dataset_prep import build_ml_dataset
from models.train_ml import train_all_models
from models.predict import load_models
from rule_engine.rules import compute_rule_signals
from features.build_features import load_feature_store


def _confusion_dict(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "true_negative": int(cm[0][0]), "false_positive": int(cm[0][1]),
        "false_negative": int(cm[1][0]), "true_positive": int(cm[1][1]),
    }


def _check_raw_data_exists(symbol: str, interval: str) -> int:
    with get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND interval = ?", (symbol, interval))
        return cur.fetchone()[0]


def _check_feature_store_exists(symbol: str, interval: str) -> int:
    with get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM feature_store WHERE symbol = ? AND interval = ?", (symbol, interval))
        return cur.fetchone()[0]


def evaluate_rule_engine(symbol: str, interval: str, test_index) -> dict:
    df = load_feature_store(symbol, interval)
    if df.empty:
        return {"note": "داده‌ای نیست"}

    rule_signals = compute_rule_signals(df)
    rule_score = rule_signals.loc[test_index, "rule_engine_score"]

    _, _, _, y_test, _, _ = build_ml_dataset(symbol, interval)

    directional = rule_score[rule_score != 0]
    if directional.empty:
        return {"note": "Rule Engine در این بازه هیچ سیگنال جهت‌داری نداده"}

    predicted_up = directional > 0
    actual_up = y_test.loc[directional.index] == 1
    win_rate = (predicted_up == actual_up).mean()

    return {
        "signals_fired": int(len(directional)), "win_rate": round(float(win_rate), 4),
        "note": "win_rate = درصد دفعاتی که جهت سیگنال با جهت واقعی قیمت یکی بود",
    }


def generate_report(symbol: str = SYMBOL, interval: str = OHLCV_INTERVAL, retrain: bool = True) -> dict:
    init_db()
    print(f"\n{'='*60}\nساخت گزارش Accuracy برای {symbol} ({interval})\n{'='*60}\n")

    raw_count = _check_raw_data_exists(symbol, interval)
    if raw_count == 0:
        raise SystemExit(
            f"\n[accuracy_report] هیچ داده خامی برای {symbol}/{interval} پیدا نشد.\n"
            f"    python main.py --init-history\n    python main.py --pipeline\n"
        )
    print(f"[accuracy_report] {raw_count} کندل خام پیدا شد.\n")

    feature_count = _check_feature_store_exists(symbol, interval)
    if feature_count == 0:
        raise SystemExit(
            f"\n[accuracy_report] داده خام هست ولی هنوز فیچری ساخته نشده.\n"
            f"    python main.py --pipeline\n"
        )
    print(f"[accuracy_report] {feature_count} ردیف فیچر پیدا شد.\n")

    if retrain:
        ml_metrics = train_all_models(symbol, interval)
    else:
        from models.train_ml import _evaluate
        models = load_models(symbol, interval)
        _, X_test, _, y_test, _, _ = build_ml_dataset(symbol, interval)
        ml_metrics = {name: _evaluate(m, X_test, y_test) for name, m in models.items()}
        majority_class_ratio = max(y_test.mean(), 1 - y_test.mean())
        ml_metrics["baseline_majority_class"] = {
            "accuracy": round(float(majority_class_ratio), 4),
            "note": "اگه مدل بهتر از این نباشه، یعنی چیزی یاد نگرفته",
        }

    models = load_models(symbol, interval)
    X_train, X_test, y_train, y_test, feature_cols, full_df = build_ml_dataset(symbol, interval)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        if name in ml_metrics:
            ml_metrics[name]["confusion_matrix"] = _confusion_dict(y_test, y_pred)

    rule_metrics = evaluate_rule_engine(symbol, interval, X_test.index)

    report = {
        "symbol": symbol, "interval": interval,
        "test_period": {
            "rows": int(len(X_test)),
            "from_open_time": int(full_df.loc[X_test.index, "open_time"].min()),
            "to_open_time": int(full_df.loc[X_test.index, "open_time"].max()),
        },
        "ml_models": ml_metrics, "rule_engine": rule_metrics,
    }
    return report


def print_report(report: dict):
    print(f"\nنماد: {report['symbol']} | تایم‌فریم: {report['interval']}")
    print(f"تعداد نمونه تست: {report['test_period']['rows']}\n")

    baseline = report["ml_models"].get("baseline_majority_class", {}).get("accuracy")
    print(f"{'مدل':<20}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<10}")
    print("-" * 66)
    for name, metrics in report["ml_models"].items():
        if name == "baseline_majority_class":
            continue
        marker = " ✓بهتر از baseline" if baseline and metrics["accuracy"] > baseline else ""
        print(f"{name:<20}{metrics['accuracy']:<12}{metrics['precision']:<12}{metrics['recall']:<12}{metrics['f1']:<10}{marker}")

    print("-" * 66)
    baseline_str = str(baseline) if baseline is not None else "نامشخص"
    print(f"{'baseline (اکثریت)':<20}{baseline_str:<12}")

    print(f"\nRule Engine:")
    for k, v in report["rule_engine"].items():
        print(f"  {k}: {v}")


def save_report(report: dict, base_dir: str = BASE_DIR):
    json_path = os.path.join(base_dir, "accuracy_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    baseline = report["ml_models"].get("baseline_majority_class", {}).get("accuracy")
    md_lines = [
        f"# گزارش Accuracy — {report['symbol']} ({report['interval']})", "",
        f"تعداد نمونه تست: **{report['test_period']['rows']}**", "",
        "| مدل | Accuracy | Precision | Recall | F1 |", "|---|---|---|---|---|",
    ]
    for name, metrics in report["ml_models"].items():
        if name == "baseline_majority_class":
            continue
        md_lines.append(f"| {name} | {metrics['accuracy']} | {metrics['precision']} | {metrics['recall']} | {metrics['f1']} |")
    md_lines.append(f"| **baseline (حدس اکثریت)** | {baseline if baseline is not None else 'نامشخص'} | - | - | - |")
    md_lines.append("")
    md_lines.append("## Rule Engine")
    for k, v in report["rule_engine"].items():
        md_lines.append(f"- **{k}**: {v}")

    md_path = os.path.join(base_dir, "accuracy_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[accuracy_report] ذخیره شد در:\n  {json_path}\n  {md_path}")


def run(symbol: str = SYMBOL, interval: str = OHLCV_INTERVAL, retrain: bool = True):
    report = generate_report(symbol, interval, retrain=retrain)
    print_report(report)
    save_report(report)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="گزارش Accuracy موتور")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--interval", default=OHLCV_INTERVAL)
    parser.add_argument("--no-retrain", action="store_true")
    args = parser.parse_args()
    run(args.symbol, args.interval, retrain=not args.no_retrain)
