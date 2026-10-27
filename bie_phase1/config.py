"""
تنظیمات کلی پروژه Bitcoin Intelligence Engine
همه چیزهایی که ممکنه بعداً بخوای تغییر بدی (نماد، تایم‌فریم، مسیر دیتابیس) اینجاست.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "bie_data.db")

# --- Binance ---
BINANCE_SPOT_BASE_URL = "https://api.binance.com"
BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
SYMBOL = "BTCUSDT"
OHLCV_INTERVAL = "1h"
OHLCV_LIMIT = 500

# --- Fear & Greed ---
FEAR_GREED_URL = "https://api.alternative.me/fng/"

# --- Macro (yfinance tickers) ---
MACRO_TICKERS = {
    "DXY": "DX-Y.NYB",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "VIX": "^VIX",
    "US10Y": "^TNX",
}

# --- News RSS ---
NEWS_RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
}

REQUEST_TIMEOUT = 15  # ثانیه

# --- فاز ۲: Validation ---
INTERVAL_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}
OUTLIER_ZSCORE_THRESHOLD = 4.0

# --- فاز ۳: Cleaning ---
MAX_CONSECUTIVE_FILL = 3

# --- فاز ۴: Feature Engineering ---
FEATURE_VERSION = "v3"  # v3: رفع نشتی داده در swing_high/low + رفع باگ تبدیل زمان macro/GDELT (v1/v2 قدیمی نشتی‌دار بودن - استفاده نکن)
EMA_WINDOWS = [9, 21, 50, 200]
SMA_WINDOWS = [20, 50]
RSI_WINDOW = 14
ATR_WINDOW = 14
ADX_WINDOW = 14
STOCH_RSI_WINDOW = 14
ROLLING_STATS_WINDOW = 20
SWING_LOOKBACK = 3
EQUAL_HL_TOLERANCE = 0.001
NEWS_WINDOWS_HOURS = [1, 6, 24]

# --- فاز ۵: ML Layer ---
MODELS_DIR = os.path.join(BASE_DIR, "models_store")
TARGET_HORIZON = 1
TEST_SIZE_RATIO = 0.2
MIN_ROWS_FOR_TRAINING = 100
NON_FEATURE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "target"]

# --- فاز ۷: Signal Aggregator ---
SIGNAL_WEIGHTS = {
    "ml": 0.5,
    "rule_engine": 0.5,
}
BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45

# --- Walk-Forward Validation ---
WALK_FORWARD_FREQ = "Q"
WALK_FORWARD_MIN_TRAIN_PERIODS = 8

# --- Historical Backfill (از ۲۰۲۰ تا الان) ---
HISTORY_START_DATE = "2020-01-01"
BINANCE_KLINES_MAX_LIMIT = 1000
BACKFILL_SLEEP_SECONDS = 0.3

# --- GDELT (اخبار تاریخی - رایگان، بدون کلید، از ۲۰۱۷ به بعد) ---
GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERY = "bitcoin"
GDELT_CHUNK_DAYS = 90

# --- Backtest ---
BACKTEST_INITIAL_CAPITAL = 10000.0
BACKTEST_FEE_RATE = 0.001
BACKTEST_SLIPPAGE_RATE = 0.0005
BACKTEST_SPREAD_RATE = 0.0005     # ۰.۰۵٪ فاصله bid/ask فرضی (نیم‌اسپرد روی هر پا اعمال می‌شه)

BACKTEST_OUTLIER_ZSCORE_CAP = 8.0
BACKTEST_OUTLIER_ROLLING_WINDOW = 90

BACKTEST_RISK_PER_TRADE = 0.01
BACKTEST_SL_ATR_MULTIPLIER = 2.0
BACKTEST_TP_ATR_MULTIPLIER = 3.0

# --- Feature Selection (SHAP) ---
FEATURE_SELECTION_TOP_N = 25
USE_SELECTED_FEATURES = False
