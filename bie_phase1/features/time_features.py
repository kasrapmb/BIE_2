"""
فیچرهای Time: Hour, Day, Session, Weekend, Month
"""

import pandas as pd


def _get_session(hour_utc: int) -> str:
    if 0 <= hour_utc < 8:
        return "asia"
    elif 8 <= hour_utc < 16:
        return "europe"
    else:
        return "us"


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    dt = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    out["hour"] = dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    out["session"] = out["hour"].apply(_get_session)

    return out
