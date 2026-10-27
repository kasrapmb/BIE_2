"""
Historical Backfill - اجرای همه‌ی منابع با یک دستور
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from config import SYMBOL, OHLCV_INTERVAL, HISTORY_START_DATE, GDELT_QUERY
from db import init_db
from historical.backfill_ohlcv import run as backfill_ohlcv
from historical.backfill_funding_rate import run as backfill_funding_rate
from historical.backfill_macro import run as backfill_macro
from historical.backfill_gdelt_news import run as backfill_gdelt_news
from collectors.fear_greed import run as collect_fear_greed


def run_all_historical(symbol: str = SYMBOL, interval: str = OHLCV_INTERVAL,
                        start_date: str = HISTORY_START_DATE, gdelt_query: str = GDELT_QUERY):
    init_db()
    print(f"\n{'='*60}\nBackfill تاریخی کامل: {symbol}/{interval} از {start_date}\n{'='*60}\n")

    print("--- ۱/۴: OHLCV (Binance) ---")
    backfill_ohlcv(symbol, interval, start_date)

    print("\n--- ۲/۴: Funding Rate (Binance) ---")
    backfill_funding_rate(symbol, start_date)

    print("\n--- ۳/۴: Fear & Greed (کل تاریخچه) ---")
    collect_fear_greed(limit=0)

    print("\n--- ۴/۴: داده‌های ماکرو (yfinance) ---")
    backfill_macro(start_date)

    print("\n--- اضافه: اخبار GDELT (کندترینه، صبر کن) ---")
    backfill_gdelt_news(gdelt_query, start_date)

    print(f"\n{'='*60}")
    print("Backfill تاریخی تمام شد. الان می‌تونی این‌ها رو اجرا کنی:")
    print("    python main.py --pipeline")
    print("    python accuracy_report.py")
    print("    python walk_forward_report.py")
    print("    python backtest_report.py")
    print(f"{'='*60}\n")
    print("یادآوری: Open Interest تاریخی جمع نشد (محدودیت رایگان Binance = ۳۰ روز اخیر).")


def run(**kwargs):
    return run_all_historical(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill تاریخی کامل")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--interval", default=OHLCV_INTERVAL)
    parser.add_argument("--start-date", default=HISTORY_START_DATE)
    parser.add_argument("--gdelt-query", default=GDELT_QUERY)
    args = parser.parse_args()
    run(symbol=args.symbol, interval=args.interval, start_date=args.start_date, gdelt_query=args.gdelt_query)
