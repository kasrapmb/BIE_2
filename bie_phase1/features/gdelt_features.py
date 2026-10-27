"""
فیچر GDELT News Volume + Tone (تاریخی، از historical/backfill_gdelt_news.py)

⚠️ نکته حیاتی درباره‌ی نشتی اطلاعات: عدد "تون روز D" خلاصه‌ی کل اخبار اون روزه.
اگه مستقیم به کندل ۰۱:۰۰ روز D وصل بشه، یعنی از اخباری که هنوز اتفاق نیفتادن
استفاده می‌شه. برای رفعش، هر عدد روزانه رو یک روز به تأخیر می‌ندازیم.

⚠️ نکته فنی: تبدیل تاریخ با Timedelta division (نه astype("int64")) - همون دلیل
macro_features.py (سازگاری با نسخه‌های مختلف pandas).
"""

import pandas as pd
from db import get_connection
from config import GDELT_QUERY

ONE_DAY_MS = 24 * 3600 * 1000


def _load_gdelt_daily(query: str) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT date, article_volume, avg_tone FROM news_sentiment_daily WHERE query = ? ORDER BY date ASC",
            conn, params=(query,),
        )
    if df.empty:
        return df
    df["date_ms"] = (pd.to_datetime(df["date"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    df["available_from_ms"] = df["date_ms"] + ONE_DAY_MS
    return df


def compute_gdelt_features(df: pd.DataFrame, query: str = GDELT_QUERY) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    gdelt_df = _load_gdelt_daily(query)

    if gdelt_df.empty:
        out["gdelt_news_volume"] = pd.NA
        out["gdelt_news_tone"] = pd.NA
        return out

    left = pd.DataFrame({"open_time": df["open_time"]}).sort_values("open_time")
    merged = pd.merge_asof(
        left,
        gdelt_df[["available_from_ms", "article_volume", "avg_tone"]].rename(columns={"available_from_ms": "open_time"}),
        on="open_time", direction="backward",
    )
    out["gdelt_news_volume"] = merged["article_volume"].values
    out["gdelt_news_tone"] = merged["avg_tone"].values
    return out
