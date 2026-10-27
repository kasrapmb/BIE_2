"""
Pipeline اصلی فازهای ۲، ۳، ۴
"""

from db import init_db
from validation.validators import run as run_validation
from cleaning.cleaners import run as run_cleaning
from features.build_features import run as run_features


def run_pipeline(symbol: str, interval: str, verbose: bool = True) -> dict:
    init_db()

    if verbose:
        print(f"\n{'='*60}")
        print(f"شروع Pipeline برای {symbol} ({interval})")
        print(f"{'='*60}")

    validation_report = run_validation(symbol, interval)
    clean_df = run_cleaning(symbol, interval)
    feature_df = run_features(symbol, interval)

    summary = {
        "symbol": symbol,
        "interval": interval,
        "validation": validation_report,
        "clean_rows": len(clean_df) if clean_df is not None else 0,
        "feature_rows": len(feature_df) if feature_df is not None else 0,
        "feature_columns": len(feature_df.columns) if feature_df is not None and not feature_df.empty else 0,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"پایان Pipeline - خلاصه:")
        print(f"  کندل خام:        {validation_report.get('total_rows', 0)}")
        print(f"  کندل تمیزشده:    {summary['clean_rows']}")
        print(f"  ردیف با فیچر:    {summary['feature_rows']}")
        print(f"  تعداد فیچر:      {summary['feature_columns']}")
        print(f"{'='*60}\n")

    return summary


if __name__ == "__main__":
    from config import SYMBOL, OHLCV_INTERVAL
    run_pipeline(SYMBOL, OHLCV_INTERVAL)
