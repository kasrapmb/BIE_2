"""
فیچرهای News: Sentiment 1H/6H/24H, News Count, Positive Ratio, Negative Ratio
sentiment ساده و keyword-based - جای مشخصی برای جایگزینی با NLP واقعی داره.
"""

import pandas as pd
from db import get_connection
from config import NEWS_WINDOWS_HOURS

POSITIVE_KEYWORDS = [
    "surge", "rally", "bullish", "gain", "soar", "record high", "adoption",
    "approve", "approval", "breakthrough", "upgrade", "partnership",
]
NEGATIVE_KEYWORDS = [
    "crash", "plunge", "bearish", "loss", "hack", "ban", "lawsuit",
    "fraud", "sell-off", "selloff", "collapse", "reject", "delay", "warning",
]


def _load_news() -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT title, published, fetched_at FROM news", conn)
    if df.empty:
        return df

    parsed = pd.to_datetime(df["published"], errors="coerce", utc=True)
    fallback = pd.to_datetime(df["fetched_at"], unit="s", utc=True)
    df["news_time"] = parsed.fillna(fallback)
    df["news_time_ms"] = (df["news_time"] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)

    title_lower = df["title"].str.lower()
    df["is_positive"] = title_lower.apply(lambda t: any(k in t for k in POSITIVE_KEYWORDS))
    df["is_negative"] = title_lower.apply(lambda t: any(k in t for k in NEGATIVE_KEYWORDS))
    return df.sort_values("news_time_ms")


def compute_news_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    news = _load_news()

    if news.empty:
        for h in NEWS_WINDOWS_HOURS:
            out[f"news_count_{h}h"] = 0
            out[f"news_positive_ratio_{h}h"] = pd.NA
            out[f"news_negative_ratio_{h}h"] = pd.NA
        return out

    open_times = df["open_time"].values
    news_times = news["news_time_ms"].values
    is_pos = news["is_positive"].values
    is_neg = news["is_negative"].values

    for h in NEWS_WINDOWS_HOURS:
        window_ms = h * 3600 * 1000
        counts, pos_ratios, neg_ratios = [], [], []
        for ot in open_times:
            in_window = (news_times <= ot) & (news_times > ot - window_ms)
            n = in_window.sum()
            counts.append(int(n))
            if n > 0:
                pos_ratios.append(float(is_pos[in_window].sum()) / n)
                neg_ratios.append(float(is_neg[in_window].sum()) / n)
            else:
                pos_ratios.append(None)
                neg_ratios.append(None)

        out[f"news_count_{h}h"] = counts
        out[f"news_positive_ratio_{h}h"] = pos_ratios
        out[f"news_negative_ratio_{h}h"] = neg_ratios

    return out
