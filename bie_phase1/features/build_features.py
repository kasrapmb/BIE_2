"""
هسته فاز ۴: Feature Store + Dataset Builder
"""

import json
import time
import pandas as pd

from config import FEATURE_VERSION
from db import get_connection
from cleaning.cleaners import load_clean_ohlcv

from features.price_features import compute_price_features
from features.candle_features import compute_candle_features
from features.trend_features import compute_trend_features
from features.momentum_features import compute_momentum_features
from features.volume_features import compute_volume_features
from features.volatility_features import compute_volatility_features
from features.statistical_features import compute_statistical_features
from features.structure_features import compute_structure_features
from features.time_features import compute_time_features
from features.fundamental_features import compute_fundamental_features
from features.macro_features import compute_macro_features
from features.news_features import compute_news_features
from features.gdelt_features import compute_gdelt_features


def build_feature_matrix(symbol: str, interval: str) -> pd.DataFrame:
    df = load_clean_ohlcv(symbol, interval)
    if df.empty:
        print(f"[features] هیچ داده تمیزی برای {symbol} ({interval}) نیست. اول فاز ۳ رو اجرا کن.")
        return df

    feature_groups = [
        compute_price_features(df),
        compute_candle_features(df),
        compute_trend_features(df),
        compute_momentum_features(df),
        compute_volume_features(df),
        compute_volatility_features(df),
        compute_statistical_features(df),
        compute_structure_features(df),
        compute_time_features(df),
        compute_fundamental_features(df, symbol),
        compute_macro_features(df),
        compute_news_features(df),
        compute_gdelt_features(df),
    ]

    base = df[["symbol", "interval", "open_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    combined = pd.concat([base] + [g.reset_index(drop=True) for g in feature_groups], axis=1)
    return combined


def save_feature_store(feature_df: pd.DataFrame, feature_version: str = FEATURE_VERSION):
    if feature_df.empty:
        print("[features] چیزی برای ذخیره در feature_store نیست.")
        return

    meta_cols = ["symbol", "interval", "open_time"]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]

    created_at = int(time.time())
    rows = []
    for _, row in feature_df.iterrows():
        features_dict = row[feature_cols].to_dict()
        clean_dict = {
            k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float)) else v))
            for k, v in features_dict.items()
        }
        rows.append((
            row["symbol"], row["interval"], int(row["open_time"]),
            feature_version, json.dumps(clean_dict), created_at,
        ))

    with get_connection() as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO feature_store
            (symbol, interval, open_time, feature_version, features_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, rows)

    print(f"[features] {len(rows)} ردیف فیچر (نسخه {feature_version}) در feature_store ذخیره شد.")


def load_feature_store(symbol: str, interval: str, feature_version: str = FEATURE_VERSION) -> pd.DataFrame:
    try:
        with get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT open_time, features_json FROM feature_store
                WHERE symbol = ? AND interval = ? AND feature_version = ?
                ORDER BY open_time ASC
            """, conn, params=(symbol, interval, feature_version))
    except Exception as e:
        print(f"[features] خطا در خواندن feature_store برای {symbol}/{interval}: {e}")
        print("[features] احتمالاً هنوز pipeline اجرا نشده. اول این دو دستور رو بزن:\n"
              "    python main.py --init-history\n"
              "    python main.py --pipeline")
        return pd.DataFrame()

    if df.empty:
        print(f"[features] هیچ رکوردی در feature_store برای {symbol}/{interval} (نسخه {feature_version}) نیست. "
              f"اول 'python main.py --init-history' و بعد 'python main.py --pipeline' رو اجرا کن.")
        return df

    features_flat = pd.json_normalize(df["features_json"].apply(json.loads))
    result = pd.concat([df[["open_time"]], features_flat], axis=1)
    return result


def build_and_save(symbol: str, interval: str, feature_version: str = FEATURE_VERSION):
    feature_df = build_feature_matrix(symbol, interval)
    save_feature_store(feature_df, feature_version)
    return feature_df


def run(symbol: str, interval: str):
    return build_and_save(symbol, interval)


if __name__ == "__main__":
    from config import SYMBOL, OHLCV_INTERVAL
    run(SYMBOL, OHLCV_INTERVAL)
