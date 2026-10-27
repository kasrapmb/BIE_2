"""
جمع‌آوری اخبار از RSS Feed های CoinDesk و Cointelegraph (لایو، اخیر)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import feedparser
from config import NEWS_RSS_FEEDS
from db import get_connection


def fetch_news(feed_url: str):
    feed = feedparser.parse(feed_url)
    return feed.entries


def save_news(source: str, entries):
    fetched_at = int(time.time())
    rows = []
    for e in entries:
        title = getattr(e, "title", "").strip()
        link = getattr(e, "link", "").strip()
        published = getattr(e, "published", "")
        if not title or not link:
            continue
        rows.append((source, title, link, published, fetched_at))

    if not rows:
        print(f"[news_rss] هیچ خبر معتبری از {source} پیدا نشد.")
        return

    with get_connection() as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO news
            (source, title, link, published, fetched_at)
            VALUES (?, ?, ?, ?, ?)
        """, rows)

    print(f"[news_rss] {len(rows)} خبر از {source} ذخیره شد.")


def run():
    for source, url in NEWS_RSS_FEEDS.items():
        try:
            entries = fetch_news(url)
            save_news(source, entries)
        except Exception as e:
            print(f"[news_rss] خطا در دریافت اخبار از {source}: {e}")


if __name__ == "__main__":
    run()
