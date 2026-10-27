"""
لایه دیتابیس - SQLite
همه‌ی داده‌های خام اینجا ذخیره می‌شن. هر منبع داده یک جدول جدا داره.
"""

import sqlite3
from contextlib import contextmanager
from config import DB_PATH


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                close_time INTEGER,
                fetched_at INTEGER NOT NULL,
                PRIMARY KEY (symbol, interval, open_time)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS fear_greed (
                timestamp INTEGER PRIMARY KEY,
                value INTEGER,
                classification TEXT,
                fetched_at INTEGER NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS funding_rate (
                symbol TEXT NOT NULL,
                funding_time INTEGER NOT NULL,
                funding_rate REAL,
                fetched_at INTEGER NOT NULL,
                PRIMARY KEY (symbol, funding_time)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS open_interest (
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open_interest REAL,
                fetched_at INTEGER NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS macro (
                ticker_name TEXT NOT NULL,
                date TEXT NOT NULL,
                close REAL,
                fetched_at INTEGER NOT NULL,
                PRIMARY KEY (ticker_name, date)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS news (
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                link TEXT NOT NULL,
                published TEXT,
                fetched_at INTEGER NOT NULL,
                PRIMARY KEY (source, link)
            )
        """)

        # --- فاز ۲: Data Validation ---
        cur.execute("""
            CREATE TABLE IF NOT EXISTS validation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                open_time INTEGER,
                issue_type TEXT NOT NULL,
                detail TEXT,
                detected_at INTEGER NOT NULL
            )
        """)

        # --- فاز ۳: Cleaning ---
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_clean (
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                close_time INTEGER,
                is_imputed INTEGER NOT NULL DEFAULT 0,
                cleaned_at INTEGER NOT NULL,
                PRIMARY KEY (symbol, interval, open_time)
            )
        """)

        # --- فاز ۴: Feature Store ---
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feature_store (
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                feature_version TEXT NOT NULL,
                features_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (symbol, interval, open_time, feature_version)
            )
        """)

        # --- Historical: اخبار روزانه از GDELT ---
        cur.execute("""
            CREATE TABLE IF NOT EXISTS news_sentiment_daily (
                query TEXT NOT NULL,
                date TEXT NOT NULL,
                article_volume REAL,
                avg_tone REAL,
                fetched_at INTEGER NOT NULL,
                PRIMARY KEY (query, date)
            )
        """)

        print(f"[db] دیتابیس آماده شد: {DB_PATH}")
