"""
جمع‌آوری داده‌های ماکرو (لایو - چند روز اخیر) از yfinance
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import yfinance as yf
from config import MACRO_TICKERS
from db import get_connection


def fetch_macro(ticker_symbol: str, period: str = "5d"):
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period=period, interval="1d")
    return hist


def save_macro(ticker_name: str, hist):
    fetched_at = int(time.time())
    rows = []
    for date_index, row in hist.iterrows():
        date_str = date_index.strftime("%Y-%m-%d")
        rows.append((ticker_name, date_str, float(row["Close"]), fetched_at))

    with get_connection() as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO macro
            (ticker_name, date, close, fetched_at)
            VALUES (?, ?, ?, ?)
        """, rows)

    print(f"[macro] {len(rows)} رکورد برای {ticker_name} ذخیره شد.")


def run(period: str = "5d"):
    for name, symbol in MACRO_TICKERS.items():
        try:
            hist = fetch_macro(symbol, period=period)
            if hist.empty:
                print(f"[macro] هشدار: داده‌ای برای {name} ({symbol}) برنگشت.")
                continue
            save_macro(name, hist)
        except Exception as e:
            print(f"[macro] خطا در دریافت {name} ({symbol}): {e}")


if __name__ == "__main__":
    run(period="1y")
