"""
جمع‌آوری کندل‌های قیمت (OHLCV) از Binance Spot API - داده لایو (آخرین N کندل)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import requests
from config import BINANCE_SPOT_BASE_URL, SYMBOL, OHLCV_INTERVAL, OHLCV_LIMIT, REQUEST_TIMEOUT
from db import get_connection


def fetch_ohlcv(symbol: str = SYMBOL, interval: str = OHLCV_INTERVAL, limit: int = OHLCV_LIMIT):
    url = f"{BINANCE_SPOT_BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def save_ohlcv(candles, symbol: str = SYMBOL, interval: str = OHLCV_INTERVAL):
    fetched_at = int(time.time())
    rows = []
    for c in candles:
        rows.append((
            symbol, interval, int(c[0]), float(c[1]), float(c[2]), float(c[3]),
            float(c[4]), float(c[5]), int(c[6]), fetched_at,
        ))

    with get_connection() as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO ohlcv
            (symbol, interval, open_time, open, high, low, close, volume, close_time, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

    print(f"[binance_ohlcv] {len(rows)} کندل برای {symbol} ({interval}) ذخیره شد.")


def run():
    try:
        candles = fetch_ohlcv()
        save_ohlcv(candles)
    except requests.exceptions.RequestException as e:
        print(f"[binance_ohlcv] خطا در دریافت داده: {e}")


if __name__ == "__main__":
    run()
