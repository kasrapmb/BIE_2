"""
گزارش Backtest - فایل مستقل برای دیدن عملکرد واقعی معاملاتی موتور
اجرا: python backtest_report.py [--freq Q/M/Y] [--min-train-periods N]

⚠️ این backtest بدون slippage واقعی صرافی و بدون نقدشوندگی کامله. نتیجه مثبت
روی داده تاریخی تضمین سود آینده نیست.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import argparse

from config import SYMBOL, OHLCV_INTERVAL, BASE_DIR, WALK_FORWARD_FREQ, WALK_FORWARD_MIN_TRAIN_PERIODS
from db import init_db, get_connection
from backtest.walk_forward_backtest import run_walk_forward_backtest


def _check_feature_store_exists(symbol: str, interval: str) -> int:
    with get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM feature_store WHERE symbol = ? AND interval = ?", (symbol, interval))
        return cur.fetchone()[0]


def print_report(result: dict):
    m = result["metrics"]
    print(f"\nنماد: {result['symbol']} | تایم‌فریم: {result['interval']} | fold ها: {result['n_folds']} (واحد {result['freq']})")
    print(f"موتور: {m.get('engine', 'نامشخص')}")
    print(f"دوره‌ی خارج از نمونه (out-of-sample): {m['periods']} کندل\n")

    print(f"{'سرمایه اولیه:':<30}${m['initial_capital']:,.2f}")
    print(f"{'سرمایه نهایی:':<30}${m['final_equity']:,.2f}")
    print(f"{'بازدهی کل استراتژی:':<30}{m['total_return_pct']}%")
    if m.get("cagr_pct") is not None:
        print(f"{'CAGR (بازدهی سالانه):':<30}{m['cagr_pct']}%")
    print(f"{'بازدهی buy & hold:':<30}{m['buy_hold_return_pct']}%")
    outperf_label = "بهتر از" if m["outperformance_pct"] > 0 else "بدتر از"
    print(f"{'نسبت به buy & hold:':<30}{m['outperformance_pct']:+}% ({outperf_label} نگه‌داشتن ساده)")
    print(f"{'حداکثر افت سرمایه:':<30}{m['max_drawdown_pct']}%")

    print(f"\n--- معاملات ---")
    print(f"{'تعداد معاملات:':<30}{m['total_trades']}")
    print(f"{'نرخ برد (PnL مثبت/کل):':<30}{m['win_rate']}")
    print(f"{'میانگین سود هر برد:':<30}${m['average_win']}")
    print(f"{'میانگین ضرر هر باخت:':<30}${m['average_loss']}")
    print(f"{'Profit Factor:':<30}{m['profit_factor']}")
    print(f"{'Expectancy هر معامله:':<30}${m['expectancy_per_trade']}")

    print(f"\n--- ریسک ---")
    print(f"{'Sharpe Ratio:':<30}{m['sharpe_ratio']}")
    print(f"{'Sortino Ratio:':<30}{m['sortino_ratio']}")
    if m.get("calmar_ratio") is not None:
        print(f"{'Calmar Ratio:':<30}{m['calmar_ratio']}")

    print(f"\n--- هزینه ---")
    print(f"{'کل کارمزد پرداختی:':<30}${m['total_fees_paid']}")
    print(f"{'کل هزینه لغزش/اسپرد:':<30}${m['total_slippage_spread_cost']}")

    if "risk_per_trade_pct" in m:
        print(f"\n--- تنظیمات ---")
        print(f"{'ریسک هر معامله:':<30}{m['risk_per_trade_pct']}% از سرمایه")
        print(f"{'ضریب SL/TP (بر پایه ATR):':<30}{m['sl_atr_multiplier']}x / {m['tp_atr_multiplier']}x")

    print(f"\n{'='*60}")
    if m["outperformance_pct"] > 0:
        print("نتیجه: روی این داده‌ی تاریخی، استراتژی بهتر از نگه‌داشتن ساده عمل کرده.")
    else:
        print("نتیجه: روی این داده‌ی تاریخی، فقط نگه‌داشتن ساده (buy & hold) بهتر بوده.")
    print("این نتیجه فقط برای این بازه‌ی زمانی خاصه - تضمینی برای آینده نیست.")
    print(f"{'='*60}")


def save_report(result: dict, base_dir: str = BASE_DIR):
    m = result["metrics"]

    json_safe = {
        "symbol": result["symbol"], "interval": result["interval"],
        "freq": result["freq"], "n_folds": result["n_folds"], "metrics": m,
    }
    json_path = os.path.join(base_dir, "backtest_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(base_dir, "backtest_equity_curve.csv")
    result["equity_curve"].to_csv(csv_path, index=False)

    trade_log_path = os.path.join(base_dir, "backtest_trade_log.json")
    if "trade_log" in result:
        with open(trade_log_path, "w", encoding="utf-8") as f:
            json.dump(result["trade_log"], f, indent=2, ensure_ascii=False)

    md_lines = [
        f"# گزارش Backtest — {result['symbol']} ({result['interval']})", "",
        f"روش: Walk-Forward Backtest ({result['n_folds']} fold، واحد {result['freq']})", "",
        "| متریک | مقدار |", "|---|---|",
        f"| سرمایه اولیه | ${m['initial_capital']:,.2f} |",
        f"| سرمایه نهایی | ${m['final_equity']:,.2f} |",
        f"| بازدهی کل | {m['total_return_pct']}% |",
    ]
    if m.get("cagr_pct") is not None:
        md_lines.append(f"| CAGR | {m['cagr_pct']}% |")
    md_lines += [
        f"| بازدهی buy & hold | {m['buy_hold_return_pct']}% |",
        f"| نسبت به buy & hold | {m['outperformance_pct']:+}% |",
        f"| حداکثر افت سرمایه | {m['max_drawdown_pct']}% |",
        f"| تعداد معاملات | {m['total_trades']} |",
        f"| نرخ برد | {m['win_rate']} |",
        f"| میانگین سود برد | ${m['average_win']} |",
        f"| میانگین ضرر باخت | ${m['average_loss']} |",
        f"| Profit Factor | {m['profit_factor']} |",
        f"| Expectancy | ${m['expectancy_per_trade']} |",
        f"| Sharpe Ratio | {m['sharpe_ratio']} |",
        f"| Sortino Ratio | {m['sortino_ratio']} |",
        f"| Calmar Ratio | {m.get('calmar_ratio')} |",
        f"| کل کارمزد | ${m['total_fees_paid']} |",
        f"| کل هزینه لغزش/اسپرد | ${m['total_slippage_spread_cost']} |",
        "",
        "## نکته مهم",
        "این یک backtest ساده‌ست (بدون slippage واقعی صرافی، بدون نقدشوندگی کامل). "
        "منحنی سرمایه کامل در `backtest_equity_curve.csv` و ریز معاملات در "
        "`backtest_trade_log.json` قابل بررسی‌ان.",
    ]
    md_path = os.path.join(base_dir, "backtest_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[backtest_report] ذخیره شد در:\n  {json_path}\n  {md_path}\n  {csv_path}\n  {trade_log_path}")


def run(symbol: str = SYMBOL, interval: str = OHLCV_INTERVAL,
        freq: str = WALK_FORWARD_FREQ, min_train_periods: int = WALK_FORWARD_MIN_TRAIN_PERIODS):
    init_db()

    feature_count = _check_feature_store_exists(symbol, interval)
    if feature_count == 0:
        raise SystemExit(
            f"\n[backtest_report] فیچری برای {symbol}/{interval} پیدا نشد.\n"
            f"    python main.py --init-history\n    python main.py --pipeline\n"
        )

    result = run_walk_forward_backtest(symbol, interval, freq=freq, min_train_periods=min_train_periods)
    print_report(result)
    save_report(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="گزارش Backtest (Walk-Forward)")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--interval", default=OHLCV_INTERVAL)
    parser.add_argument("--freq", default=WALK_FORWARD_FREQ, choices=["Q", "M", "Y"])
    parser.add_argument("--min-train-periods", type=int, default=WALK_FORWARD_MIN_TRAIN_PERIODS)
    args = parser.parse_args()
    run(args.symbol, args.interval, freq=args.freq, min_train_periods=args.min_train_periods)
