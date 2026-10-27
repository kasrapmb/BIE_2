"""
Historical Backfill: OHLCV از HISTORY_START_DATE تا الان (paginated از Binance)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import datetime
import requests

from config import (
    BINANCE_SPOT_BASE_URL, SYMBOL, OHLCV_INTERVAL, INTERVAL_MS,
    HISTORY_START_DATE, BINANCE_KLINES_MAX_LIMIT, BACKFILL_SLEEP_SECONDS, REQUEST_TIMEOUT,
)
from db import init_db, get_connection


def _date_to_ms(date_str: str) -> int:
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines_page(symbol: str, interval: str, start_ms: int, limit: int = BINANCE_KLINES_MAX_LIMIT):
    url = f"{BINANCE_SPOT_BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": limit}
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def save_klines_page(candles, symbol: str, interval: str):
    if not candles:
        return 0
    fetched_at = int(time.time())
    rows = [
        (symbol, interval, int(c[0]), float(c[1]), float(c[2]), float(c[3]),
         float(c[4]), float(c[5]), int(c[6]), fetched_at)
        for c in candles
    ]
    with get_connection() as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO ohlcv
            (symbol, interval, open_time, open, high, low, close, volume, close_time, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
    return len(rows)


def backfill_ohlcv(symbol: str = SYMBOL, interval: str = OHLCV_INTERVAL,
                    start_date: str = HISTORY_START_DATE) -> int:
    init_db()
    step_ms = INTERVAL_MS.get(interval)
    if step_ms is None:
        raise ValueError(f"interval ناشناخته: {interval}")

    current_ms = _date_to_ms(start_date)
    now_ms = int(time.time() * 1000)
    total_saved = 0
    page = 0

    print(f"[backfill_ohlcv] شروع جمع‌آوری {symbol} ({interval}) از {start_date} تا الان...")

    while current_ms < now_ms:
        page += 1
        try:
            candles = fetch_klines_page(symbol, interval, current_ms)
        except requests.exceptions.RequestException as e:
            print(f"[backfill_ohlcv] خطا در صفحه {page}: {e} - skip و ادامه.")
            current_ms += step_ms * BINANCE_KLINES_MAX_LIMIT
            time.sleep(BACKFILL_SLEEP_SECONDS)
            continue

        if not candles:
            break

        saved = save_klines_page(candles, symbol, interval)
        total_saved += saved

        last_open_time = int(candles[-1][0])
        last_date = datetime.datetime.fromtimestamp(last_open_time / 1000, tz=datetime.timezone.utc).date()
        print(f"[backfill_ohlcv] صفحه {page}: {saved} کندل ذخیره شد (تا {last_date})")

        current_ms = last_open_time + step_ms
        time.sleep(BACKFILL_SLEEP_SECONDS)

        if len(candles) < BINANCE_KLINES_MAX_LIMIT:
            break

    print(f"[backfill_ohlcv] تمام شد. مجموعاً {total_saved} کندل برای {symbol}/{interval} ذخیره شد.")
    return total_saved


def run(symbol: str = SYMBOL, interval: str = OHLCV_INTERVAL, start_date: str = HISTORY_START_DATE):
    return backfill_ohlcv(symbol, interval, start_date)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backfill تاریخی OHLCV از Binance")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--interval", default=OHLCV_INTERVAL)
    parser.add_argument("--start-date", default=HISTORY_START_DATE)
    args = parser.parse_args()
    run(args.symbol, args.interval, args.start_date)
