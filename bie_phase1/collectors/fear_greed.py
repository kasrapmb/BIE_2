"""
جمع‌آوری شاخص ترس و طمع (Fear & Greed Index) - رایگان، بدون نیاز به API key
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import requests
from config import FEAR_GREED_URL, REQUEST_TIMEOUT
from db import get_connection


def fetch_fear_greed(limit: int = 30):
    params = {"limit": limit}
    response = requests.get(FEAR_GREED_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return data.get("data", [])


def save_fear_greed(entries):
    fetched_at = int(time.time())
    rows = []
    for e in entries:
        rows.append((int(e["timestamp"]), int(e["value"]), e["value_classification"], fetched_at))

    with get_connection() as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO fear_greed
            (timestamp, value, classification, fetched_at)
            VALUES (?, ?, ?, ?)
        """, rows)

    print(f"[fear_greed] {len(rows)} رکورد ذخیره شد.")


def run(limit: int = 30):
    try:
        entries = fetch_fear_greed(limit=limit)
        save_fear_greed(entries)
    except requests.exceptions.RequestException as e:
        print(f"[fear_greed] خطا در دریافت داده: {e}")


if __name__ == "__main__":
    run(limit=30)
