"""
فیچرهای Macro: DXY, S&P500, Nasdaq, VIX, US10Y
داده‌های ماکرو روزانه‌ان، پس با merge_asof هر کندل آخرین close معتبر رو می‌گیره.

⚠️ نکته حیاتی درباره‌ی نشتی اطلاعات: مقدار close بازار روز D فقط *آخر* همون روز
مشخص می‌شه، نه اول روز. برای رفعش، هر مقدار روزانه رو یک روز به تأخیر می‌ندازیم.

⚠️ نکته فنی مهم: تبدیل تاریخ به میلی‌ثانیه با Timedelta division انجام می‌شه
(نه .astype("int64")) چون واحد داخلی datetime64 در نسخه‌های مختلف pandas فرق
می‌کنه (نانوثانیه/میکروثانیه/ثانیه) و astype("int64") به این واحد وابسته‌ست -
Timedelta division همیشه درست کار می‌کنه، مستقل از نسخه‌ی pandas.
"""

import pandas as pd
from db import get_connection
from config import MACRO_TICKERS

ONE_DAY_MS = 24 * 3600 * 1000


def _load_macro_ticker(ticker_name: str) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT date, close FROM macro WHERE ticker_name = ? ORDER BY date ASC",
            conn, params=(ticker_name,),
        )
    if df.empty:
        return df
    df["date_ms"] = (pd.to_datetime(df["date"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    df["available_from_ms"] = df["date_ms"] + ONE_DAY_MS
    return df


def compute_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    left = pd.DataFrame({"open_time": df["open_time"]}).sort_values("open_time")

    for name in MACRO_TICKERS.keys():
        macro_df = _load_macro_ticker(name)
        col = f"macro_{name.lower()}"
        if macro_df.empty:
            out[col] = pd.NA
            continue
        merged = pd.merge_asof(
            left,
            macro_df[["available_from_ms", "close"]].rename(columns={"available_from_ms": "open_time"}),
            on="open_time", direction="backward",
        )
        out[col] = merged["close"].values

    return out
