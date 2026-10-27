"""
Export نهایی Dataset - فیچرهای ذخیره‌شده در feature_store رو به CSV تبدیل می‌کنه
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from features.build_features import load_feature_store
from config import SYMBOL, OHLCV_INTERVAL, FEATURE_VERSION, BASE_DIR


def export_dataset(symbol: str, interval: str, feature_version: str = FEATURE_VERSION, output_path: str = None):
    df = load_feature_store(symbol, interval, feature_version)
    if df.empty:
        print(f"[export] هیچ فیچری برای {symbol} ({interval}, نسخه {feature_version}) پیدا نشد.")
        return None

    if output_path is None:
        output_path = os.path.join(BASE_DIR, f"dataset_{symbol}_{interval}_{feature_version}.csv")

    df.to_csv(output_path, index=False)
    print(f"[export] دیتاست نهایی ذخیره شد: {output_path}")
    print(f"[export] شکل دیتاست: {df.shape[0]} ردیف × {df.shape[1]} ستون")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export دیتاست نهایی از Feature Store")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--interval", default=OHLCV_INTERVAL)
    parser.add_argument("--version", default=FEATURE_VERSION)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    export_dataset(args.symbol, args.interval, args.version, args.output)
