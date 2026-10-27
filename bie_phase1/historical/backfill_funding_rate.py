"""
Historical Backfill: Funding Rate از HISTORY_START_DATE تا الان
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import datetime
import requests

from config import BINANCE_FUTURES_BASE_URL, SYMBOL, HISTORY_START_DATE, BACKFILL_SLEEP_SECONDS, REQUEST_TIMEOUT
from db import init_db, get_connection

PAGE_LIMIT = 1000


def _date_to_ms(date_str: str) -> int:
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_funding_page(symbol: str, start_ms: int, limit: int = PAGE_LIMIT):
    url = f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/fundingRate"
    params = {"symbol": symbol, "startTime": start_ms, "limit": limit}
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def save_funding_page(entries, symbol: str):
    if not entries:
        return 0
    fetched_at = int(time.time())
    rows = [(symbol, int(e["fundingTime"]), float(e["fundingRate"]), fetched_at) for e in entries]
    with get_connection() as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO funding_rate (symbol, funding_time, funding_rate, fetched_at)
            VALUES (?, ?, ?, ?)
        """, rows)
    return len(rows)


def backfill_funding_rate(symbol: str = SYMBOL, start_date: str = HISTORY_START_DATE) -> int:
    init_db()
    current_ms = _date_to_ms(start_date)
    now_ms = int(time.time() * 1000)
    total_saved = 0
    page = 0

    print(f"[backfill_funding_rate] شروع جمع‌آوری {symbol} از {start_date} تا الان...")

    while current_ms < now_ms:
        page += 1
        try:
            entries = fetch_funding_page(symbol, current_ms)
        except requests.exceptions.RequestException as e:
            print(f"[backfill_funding_rate] خطا در صفحه {page}: {e} - عبور از این بازه.")
            current_ms += 8 * 3600 * 1000 * PAGE_LIMIT
            time.sleep(BACKFILL_SLEEP_SECONDS)
            continue

        if not entries:
            break

        saved = save_funding_page(entries, symbol)
        total_saved += saved

        last_time = int(entries[-1]["fundingTime"])
        last_date = datetime.datetime.fromtimestamp(last_time / 1000, tz=datetime.timezone.utc).date()
        print(f"[backfill_funding_rate] صفحه {page}: {saved} رکورد ذخیره شد (تا {last_date})")

        current_ms = last_time + 1
        time.sleep(BACKFILL_SLEEP_SECONDS)

        if len(entries) < PAGE_LIMIT:
            break

    print(f"[backfill_funding_rate] تمام شد. مجموعاً {total_saved} رکورد برای {symbol} ذخیره شد.")
    return total_saved


def run(symbol: str = SYMBOL, start_date: str = HISTORY_START_DATE):
    return backfill_funding_rate(symbol, start_date)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backfill تاریخی Funding Rate از Binance")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--start-date", default=HISTORY_START_DATE)
    args = parser.parse_args()
    run(args.symbol, args.start_date)
