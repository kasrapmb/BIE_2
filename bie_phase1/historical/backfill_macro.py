"""
Historical Backfill: داده‌های ماکرو از HISTORY_START_DATE تا الان (yfinance)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import yfinance as yf

from config import MACRO_TICKERS, HISTORY_START_DATE
from db import init_db, get_connection


def fetch_macro_history(ticker_symbol: str, start_date: str):
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(start=start_date, interval="1d")
    return hist


def save_macro_history(ticker_name: str, hist):
    if hist.empty:
        return 0
    fetched_at = int(time.time())
    rows = []
    for date_index, row in hist.iterrows():
        date_str = date_index.strftime("%Y-%m-%d")
        rows.append((ticker_name, date_str, float(row["Close"]), fetched_at))

    with get_connection() as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO macro (ticker_name, date, close, fetched_at)
            VALUES (?, ?, ?, ?)
        """, rows)
    return len(rows)


def backfill_macro(start_date: str = HISTORY_START_DATE) -> dict:
    init_db()
    results = {}
    for name, symbol in MACRO_TICKERS.items():
        print(f"[backfill_macro] در حال گرفتن {name} ({symbol}) از {start_date}...")
        try:
            hist = fetch_macro_history(symbol, start_date)
            saved = save_macro_history(name, hist)
            results[name] = saved
            print(f"[backfill_macro] {name}: {saved} روز ذخیره شد.")
        except Exception as e:
            print(f"[backfill_macro] خطا در {name} ({symbol}): {e}")
            results[name] = 0
    return results


def run(start_date: str = HISTORY_START_DATE):
    return backfill_macro(start_date)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backfill تاریخی داده‌های ماکرو از yfinance")
    parser.add_argument("--start-date", default=HISTORY_START_DATE)
    args = parser.parse_args()
    run(args.start_date)
