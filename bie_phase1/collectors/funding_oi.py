"""
جمع‌آوری Funding Rate و Open Interest از Binance Futures API (داده لایو)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import requests
from config import BINANCE_FUTURES_BASE_URL, SYMBOL, REQUEST_TIMEOUT
from db import get_connection


def fetch_funding_rate(symbol: str = SYMBOL, limit: int = 100):
    url = f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_open_interest(symbol: str = SYMBOL):
    url = f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/openInterest"
    params = {"symbol": symbol}
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def save_funding_rate(entries, symbol: str = SYMBOL):
    fetched_at = int(time.time())
    rows = [(symbol, int(e["fundingTime"]), float(e["fundingRate"]), fetched_at) for e in entries]

    with get_connection() as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO funding_rate
            (symbol, funding_time, funding_rate, fetched_at)
            VALUES (?, ?, ?, ?)
        """, rows)

    print(f"[funding_oi] {len(rows)} رکورد funding rate ذخیره شد.")


def save_open_interest(data, symbol: str = SYMBOL):
    fetched_at = int(time.time())
    row = (symbol, int(data["time"]), float(data["openInterest"]), fetched_at)

    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO open_interest
            (symbol, timestamp, open_interest, fetched_at)
            VALUES (?, ?, ?, ?)
        """, row)

    print(f"[funding_oi] open interest لحظه‌ای ذخیره شد: {data['openInterest']}")


def run():
    try:
        funding_entries = fetch_funding_rate()
        save_funding_rate(funding_entries)
    except requests.exceptions.RequestException as e:
        print(f"[funding_oi] خطا در دریافت funding rate: {e}")

    try:
        oi_data = fetch_open_interest()
        save_open_interest(oi_data)
    except requests.exceptions.RequestException as e:
        print(f"[funding_oi] خطا در دریافت open interest: {e}")


if __name__ == "__main__":
    run()
