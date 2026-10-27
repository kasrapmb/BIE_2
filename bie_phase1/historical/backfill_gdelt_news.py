"""
Historical Backfill: حجم + تون روزانه‌ی اخبار مرتبط با بیت‌کوین، از GDELT Project
رایگان، بدون کلید، از ۲۰۱۷ به بعد. فقط حجم/تون روزانه می‌ده، نه متن کامل مقاله.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import datetime
import requests

from config import GDELT_BASE_URL, GDELT_QUERY, GDELT_CHUNK_DAYS, HISTORY_START_DATE, REQUEST_TIMEOUT
from db import init_db, get_connection


def _fmt(dt: datetime.datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def fetch_gdelt_chunk(query: str, start: datetime.datetime, end: datetime.datetime, mode: str) -> dict:
    """
    mode باید 'timelinevol' (حجم) یا 'timelinetone' (تون) باشه - GDELT این دو رو
    در دو endpoint جدا می‌ده.
    """
    params = {
        "query": query, "mode": mode, "format": "json",
        "startdatetime": _fmt(start), "enddatetime": _fmt(end),
    }
    response = requests.get(GDELT_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def parse_gdelt_response(raw: dict) -> list:
    results = []
    timeline = raw.get("timeline", [])
    for series in timeline:
        for point in series.get("data", []):
            date_raw = point.get("date", "")
            value = point.get("value")
            if not date_raw or value is None:
                continue
            try:
                if len(date_raw) >= 8 and date_raw[:8].isdigit():
                    date_str = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
                else:
                    date_str = date_raw[:10]
                results.append((date_str, float(value)))
            except (ValueError, IndexError):
                continue
    return results


def save_gdelt_points(query: str, points: list, column: str):
    if not points:
        return 0
    if column not in ("article_volume", "avg_tone"):
        raise ValueError(f"ستون نامعتبر: {column}")

    fetched_at = int(time.time())
    saved = 0
    with get_connection() as conn:
        for date_str, value in points:
            conn.execute(f"""
                INSERT INTO news_sentiment_daily (query, date, {column}, fetched_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(query, date) DO UPDATE SET {column} = excluded.{column}, fetched_at = excluded.fetched_at
            """, (query, date_str, value, fetched_at))
            saved += 1
    return saved


def backfill_gdelt_news(query: str = GDELT_QUERY, start_date: str = HISTORY_START_DATE) -> int:
    init_db()
    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    now_dt = datetime.datetime.now(datetime.timezone.utc)

    total_saved = 0
    chunk_start = start_dt
    chunk_num = 0

    print(f"[backfill_gdelt] شروع جمع‌آوری اخبار '{query}' از {start_date} تا الان...")

    while chunk_start < now_dt:
        chunk_num += 1
        chunk_end = min(chunk_start + datetime.timedelta(days=GDELT_CHUNK_DAYS), now_dt)

        chunk_saved = 0
        for mode, column in [("timelinevol", "article_volume"), ("timelinetone", "avg_tone")]:
            try:
                raw = fetch_gdelt_chunk(query, chunk_start, chunk_end, mode=mode)
                points = parse_gdelt_response(raw)
                chunk_saved += save_gdelt_points(query, points, column)
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                print(f"[backfill_gdelt] خطا در تکه {chunk_num} ({mode}): {e} - skip.")
            except Exception as e:
                print(f"[backfill_gdelt] خطای غیرمنتظره در تکه {chunk_num} ({mode}): {e} - skip.")

        total_saved += chunk_saved
        print(f"[backfill_gdelt] تکه {chunk_num} ({chunk_start.date()} تا {chunk_end.date()}): {chunk_saved} نقطه ذخیره شد.")

        chunk_start = chunk_end
        time.sleep(1)

    print(f"[backfill_gdelt] تمام شد. مجموعاً {total_saved} روز داده برای '{query}' ذخیره شد.")
    print(f"[backfill_gdelt] یادآوری: این فقط حجم/تون روزانه‌ست، نه متن کامل اخبار.")
    return total_saved


def run(query: str = GDELT_QUERY, start_date: str = HISTORY_START_DATE):
    return backfill_gdelt_news(query, start_date)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backfill تاریخی حجم/تون اخبار از GDELT")
    parser.add_argument("--query", default=GDELT_QUERY)
    parser.add_argument("--start-date", default=HISTORY_START_DATE)
    args = parser.parse_args()
    run(args.query, args.start_date)
