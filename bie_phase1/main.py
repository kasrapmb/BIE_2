"""
نقطه ورود اصلی پروژه - فازهای ۱ تا ۹ + Historical + Backtest

۱) فقط جمع‌آوری داده خام (فاز ۱) - یک‌باره:
    python main.py --once
۲) اجرای اولیه با تاریخچه کامل‌تر (فاز ۱):
    python main.py --init-history
۳) اجرای فازهای ۲-۴ (validation + cleaning + features):
    python main.py --pipeline
۴) آموزش مدل‌های ML (فاز ۵):
    python main.py --train
۵) گزارش accuracy کامل:
    python main.py --accuracy
۶) گزارش Walk-Forward Validation:
    python main.py --walkforward
۶.۵) Backfill تاریخی کامل از ۲۰۲۰ (OHLCV+Funding+Fear&Greed+Macro+GDELT):
    python main.py --historical
۶.۶) گزارش Backtest (شبیه‌سازی معاملاتی واقعی با SL/TP دینامیک):
    python main.py --backtest
۶.۷) Feature Selection با SHAP:
    python main.py --select-features
۷) همه چیز با هم، یک‌جا:
    python main.py --once --pipeline --train --accuracy --walkforward
۸) اجرای مداوم با زمان‌بندی خودکار:
    python main.py
۹) اجرای API (جدا):
    uvicorn api.main:app --reload --port 8000
"""

import argparse
import time
import schedule

from db import init_db
from config import SYMBOL, OHLCV_INTERVAL
from collectors import binance_ohlcv, fear_greed, funding_oi, macro, news_rss
from pipeline import run_pipeline
from models.train_ml import run as train_models
from models.feature_selection import run as run_feature_selection
from accuracy_report import run as run_accuracy_report
from walk_forward_report import run as run_walk_forward_report
from historical.run_all import run as run_historical_backfill
from backtest_report import run as run_backtest_report


def run_all():
    print("\n=== شروع جمع‌آوری داده ===")
    binance_ohlcv.run()
    fear_greed.run(limit=30)
    funding_oi.run()
    news_rss.run()
    macro.run(period="5d")
    print("=== پایان جمع‌آوری داده ===\n")


def run_once():
    print("\n=== اجرای اولیه (پر کردن تاریخچه) ===")
    binance_ohlcv.run()
    fear_greed.run(limit=0)
    funding_oi.run()
    news_rss.run()
    macro.run(period="1y")
    print("=== پایان اجرای اولیه ===\n")


def start_scheduler():
    schedule.every(1).hours.do(binance_ohlcv.run)
    schedule.every(1).hours.do(funding_oi.run)
    schedule.every(6).hours.do(lambda: fear_greed.run(limit=5))
    schedule.every(6).hours.do(news_rss.run)
    schedule.every().day.at("00:10").do(lambda: macro.run(period="5d"))
    schedule.every(1).hours.do(lambda: run_pipeline(SYMBOL, OHLCV_INTERVAL))

    print("[scheduler] زمان‌بندی شروع شد. برای توقف Ctrl+C بزن.")
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bitcoin Intelligence Engine - Phase 1-9 + Historical + Backtest")
    parser.add_argument("--once", action="store_true", help="فقط یک‌بار جمع‌آوری خام (فاز ۱) و خروج")
    parser.add_argument("--init-history", action="store_true", help="اجرای اولیه با تاریخچه کامل‌تر (فاز ۱)")
    parser.add_argument("--pipeline", action="store_true", help="اجرای فازهای ۲-۴")
    parser.add_argument("--train", action="store_true", help="آموزش مدل‌های ML (فاز ۵)")
    parser.add_argument("--accuracy", action="store_true", help="ساخت گزارش accuracy کامل")
    parser.add_argument("--walkforward", action="store_true", help="ساخت گزارش Walk-Forward Validation")
    parser.add_argument("--historical", action="store_true", help="Backfill تاریخی کامل از ۲۰۲۰ (کند)")
    parser.add_argument("--backtest", action="store_true", help="ساخت گزارش Backtest")
    parser.add_argument("--select-features", action="store_true", help="Feature Selection با SHAP")
    args = parser.parse_args()

    init_db()

    if args.historical:
        run_historical_backfill(symbol=SYMBOL, interval=OHLCV_INTERVAL)

    if args.init_history:
        run_once()
    elif args.once:
        run_all()

    if args.pipeline:
        run_pipeline(SYMBOL, OHLCV_INTERVAL)

    if args.select_features:
        run_feature_selection(SYMBOL, OHLCV_INTERVAL)

    if args.train:
        train_models(SYMBOL, OHLCV_INTERVAL)

    if args.accuracy:
        run_accuracy_report(SYMBOL, OHLCV_INTERVAL, retrain=True)

    if args.walkforward:
        run_walk_forward_report(SYMBOL, OHLCV_INTERVAL)

    if args.backtest:
        run_backtest_report(SYMBOL, OHLCV_INTERVAL)

    if not any([args.once, args.init_history, args.pipeline, args.train, args.accuracy,
                args.walkforward, args.historical, args.backtest, args.select_features]):
        run_all()
        run_pipeline(SYMBOL, OHLCV_INTERVAL)
        start_scheduler()
