"""
گزارش Walk-Forward Validation - فایل مستقل
اجرا: python walk_forward_report.py [--freq Q/M/Y] [--min-train-periods N]
"""

import os
import json
import argparse

from config import SYMBOL, OHLCV_INTERVAL, BASE_DIR, WALK_FORWARD_FREQ, WALK_FORWARD_MIN_TRAIN_PERIODS
from db import init_db, get_connection
from models.walk_forward import run_walk_forward


def _check_feature_store_exists(symbol: str, interval: str) -> int:
    with get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM feature_store WHERE symbol = ? AND interval = ?", (symbol, interval))
        return cur.fetchone()[0]


def print_report(result: dict):
    print(f"\nنماد: {result['symbol']} | تایم‌فریم: {result['interval']} | واحد fold: {result['freq']}")
    print(f"تعداد fold: {result['n_folds']}\n")

    print(f"{'دوره تست':<12}", end="")
    model_names = list(result["folds"][0].keys())
    model_names = [k for k in model_names if k not in ("fold", "test_period", "train_rows", "test_rows")]
    for name in model_names:
        print(f"{name+'(acc)':<18}", end="")
    print()
    print("-" * (12 + 18 * len(model_names)))

    for f in result["folds"]:
        print(f"{f['test_period']:<12}", end="")
        for name in model_names:
            print(f"{f[name]['accuracy']:<18}", end="")
        print()

    print(f"\n{'='*60}\nمیانگین روی همه fold ها\n{'='*60}")
    print(f"{'مدل':<18}{'Accuracy':<20}{'Precision':<20}{'Recall':<20}{'F1':<20}")
    print("-" * 98)
    for name, s in result["summary"].items():
        acc_str = f"{s['mean_accuracy']} ± {s['std_accuracy']}"
        prec_str = f"{s['mean_precision']} ± {s['std_precision']}"
        rec_str = f"{s['mean_recall']} ± {s['std_recall']}"
        f1_str = f"{s['mean_f1']} ± {s['std_f1']}"
        print(f"{name:<18}{acc_str:<20}{prec_str:<20}{rec_str:<20}{f1_str:<20}")

    print(f"\nRule Engine:")
    rs = result["rule_engine_summary"]
    print(f"  fold هایی که سیگنال داشتن: {rs['folds_with_signal']} از {result['n_folds']}")
    if rs["mean_win_rate"] is not None:
        print(f"  میانگین win_rate: {rs['mean_win_rate']} ± {rs['std_win_rate']}")


def save_report(result: dict, base_dir: str = BASE_DIR):
    json_path = os.path.join(base_dir, "walk_forward_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    md_lines = [
        f"# گزارش Walk-Forward Validation — {result['symbol']} ({result['interval']})", "",
        f"واحد هر fold: **{result['freq']}** | تعداد fold: **{result['n_folds']}**", "",
        "## میانگین روی همه fold ها", "",
        "| مدل | Accuracy | Precision | Recall | F1 |", "|---|---|---|---|---|",
    ]
    for name, s in result["summary"].items():
        md_lines.append(
            f"| {name} | {s['mean_accuracy']} ± {s['std_accuracy']} "
            f"| {s['mean_precision']} ± {s['std_precision']} "
            f"| {s['mean_recall']} ± {s['std_recall']} "
            f"| {s['mean_f1']} ± {s['std_f1']} |"
        )

    md_lines += ["", "## جزئیات هر Fold", "", "| Fold | دوره تست | Train rows | Test rows |"]
    for name in result["summary"].keys():
        md_lines[-1] += f" {name} Acc |"
    md_lines.append("|---|---|---|---|" + "---|" * len(result["summary"]))
    for f in result["folds"]:
        row = f"| {f['fold']} | {f['test_period']} | {f['train_rows']} | {f['test_rows']} |"
        for name in result["summary"].keys():
            row += f" {f[name]['accuracy']} |"
        md_lines.append(row)

    rs = result["rule_engine_summary"]
    md_lines += [
        "", "## Rule Engine",
        f"- fold های دارای سیگنال: {rs['folds_with_signal']} از {result['n_folds']}",
        f"- میانگین win_rate: {rs['mean_win_rate']} ± {rs['std_win_rate']}" if rs["mean_win_rate"] is not None else "- هیچ سیگنالی صادر نشد",
    ]

    md_path = os.path.join(base_dir, "walk_forward_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[walk_forward_report] ذخیره شد در:\n  {json_path}\n  {md_path}")


def run(symbol: str = SYMBOL, interval: str = OHLCV_INTERVAL,
        freq: str = WALK_FORWARD_FREQ, min_train_periods: int = WALK_FORWARD_MIN_TRAIN_PERIODS):
    init_db()

    feature_count = _check_feature_store_exists(symbol, interval)
    if feature_count == 0:
        raise SystemExit(
            f"\n[walk_forward_report] فیچری برای {symbol}/{interval} پیدا نشد.\n"
            f"    python main.py --init-history\n    python main.py --pipeline\n"
        )

    result = run_walk_forward(symbol, interval, freq=freq, min_train_periods=min_train_periods)
    print_report(result)
    save_report(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="گزارش Walk-Forward Validation")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--interval", default=OHLCV_INTERVAL)
    parser.add_argument("--freq", default=WALK_FORWARD_FREQ, choices=["Q", "M", "Y"])
    parser.add_argument("--min-train-periods", type=int, default=WALK_FORWARD_MIN_TRAIN_PERIODS)
    args = parser.parse_args()
    run(args.symbol, args.interval, freq=args.freq, min_train_periods=args.min_train_periods)
