"""
Backtest Engine (نسخه ۲ - استاندارد و قابل‌دفاع)
دو موتور:
  - run_backtest (ساده، legacy): همیشه یا ۱۰۰٪ لانگ یا فلت - فقط برای مقایسه سریع.
  - run_backtest_advanced: مدیریت سرمایه واقعی، SL/TP بر پایه ATR، معاملات trade-level
    با محاسبه‌ی دقیق PnL، و امکان مقایسه‌ی compounding در برابر fixed-size.

نکات طراحی کلیدی (طبق چک‌لیست استاندارد backtest):
  - Entry: سیگنال در کندل i، اجرا با open کندل i+1 (نه close همون کندل).
  - SL/TP: بر پایه‌ی ATR لحظه‌ی سیگنال (نه درصد ثابت)؛ اگه هم SL هم TP توی یک کندل
    لمس بشن، محافظه‌کارانه فرض می‌کنیم SL زودتر خورده.
  - Position Sizing: "ریسک ۱٪" یعنی حداکثر ضرر در SL = ۱٪ سرمایه - نه اینکه حجم
    پوزیشن ۱٪ سرمایه باشه.
  - هزینه: کارمزد + لغزش (slippage) + نیم‌اسپرد، روی هر ورود و خروج، قبل از PnL خالص.
  - Win Rate: معاملات با PnL خالص مثبت ÷ کل معاملات بسته‌شده (نه TP-hit ÷ کل خروج‌ها).
  - دو حالت گزارش جدا: compounding در برابر fixed - تا معلوم بشه سود از کجا میاد.

⚠️ محافظت در برابر داده خراب (sanitize_ohlc_for_backtest): قبل از هر محاسبه‌ای،
کندل‌های با جهش قیمتی غیرعادی شناسایی و خنثی می‌شن.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import (
    BACKTEST_INITIAL_CAPITAL, BACKTEST_FEE_RATE, BACKTEST_SLIPPAGE_RATE, BACKTEST_SPREAD_RATE,
    BACKTEST_OUTLIER_ZSCORE_CAP, BACKTEST_OUTLIER_ROLLING_WINDOW,
    BACKTEST_RISK_PER_TRADE, BACKTEST_SL_ATR_MULTIPLIER, BACKTEST_TP_ATR_MULTIPLIER,
)

MS_PER_YEAR = 365.25 * 24 * 3600 * 1000


def sanitize_ohlc_for_backtest(df: pd.DataFrame,
                                zscore_cap: float = BACKTEST_OUTLIER_ZSCORE_CAP,
                                rolling_window: int = BACKTEST_OUTLIER_ROLLING_WINDOW) -> pd.DataFrame:
    """
    کندل‌هایی که بازدهی‌شون نسبت به نوسان *اخیر* خودشون (rolling) بیش‌ازحد غیرعادیه
    رو به‌عنوان "بدون حرکت" در نظر می‌گیره (نه حذف - چون حذف شبکه‌ی زمانی رو به‌هم می‌زنه).
    """
    df = df.copy().reset_index(drop=True)
    log_return = np.log(df["close"] / df["close"].shift(1))

    rolling_mean = log_return.rolling(rolling_window, min_periods=10).mean()
    rolling_std = log_return.rolling(rolling_window, min_periods=10).std()

    z = (log_return - rolling_mean) / rolling_std.replace(0, np.nan)
    suspicious = z.abs() > zscore_cap
    suspicious = suspicious.fillna(False)

    n_suspicious = int(suspicious.sum())
    if n_suspicious > 0:
        suspicious_times = df.loc[suspicious, "open_time"].tolist() if "open_time" in df.columns else []
        print(f"[backtest][هشدار] {n_suspicious} کندل با جهش قیمتی غیرعادی (z-score > {zscore_cap}) "
              f"پیدا شد و به‌عنوان 'بدون حرکت' خنثی شد.")
        if suspicious_times:
            print(f"[backtest][هشدار] open_time این کندل‌ها: {suspicious_times[:10]}"
                  f"{' ...' if len(suspicious_times) > 10 else ''}")
        prev_close = df["close"].shift(1)
        for col in ["open", "high", "low", "close"]:
            df.loc[suspicious, col] = prev_close[suspicious]

    return df


def signals_to_positions(final_signal: pd.Series) -> pd.Series:
    position = pd.Series(0.0, index=final_signal.index)
    position[final_signal == "BUY"] = 1.0
    position[final_signal == "SELL"] = 0.0
    hold_mask = final_signal == "HOLD"
    position[hold_mask] = np.nan
    position = position.ffill().fillna(0.0)
    return position


def run_backtest(df: pd.DataFrame, final_signal: pd.Series,
                  initial_capital: float = BACKTEST_INITIAL_CAPITAL,
                  fee_rate: float = BACKTEST_FEE_RATE) -> dict:
    """موتور ساده (legacy): همیشه یا کاملاً لانگ یا کاملاً فلت."""
    df = sanitize_ohlc_for_backtest(df).reset_index(drop=True)
    signal = final_signal.reset_index(drop=True)

    close = df["close"]
    open_time = df["open_time"]

    position = signals_to_positions(signal)
    position_applied = position.shift(1).fillna(0.0)

    market_return = close.pct_change().fillna(0.0)
    strategy_return_gross = position_applied * market_return

    position_change = position_applied.diff().abs().fillna(position_applied.abs())
    fees = position_change * fee_rate
    strategy_return_net = strategy_return_gross - fees

    equity = initial_capital * (1 + strategy_return_net).cumprod()
    buy_hold_equity = initial_capital * (1 + market_return).cumprod()

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    total_trades = int((position_change > 0).sum())
    total_return = float(equity.iloc[-1] / initial_capital - 1)
    buy_hold_return = float(buy_hold_equity.iloc[-1] / initial_capital - 1)

    ret_std = strategy_return_net.std()
    sharpe_like = float(strategy_return_net.mean() / ret_std) if ret_std and ret_std > 0 else 0.0

    winning_periods = int((strategy_return_net > 0).sum())
    active_periods = int((position_applied != 0).sum())
    win_rate = round(winning_periods / active_periods, 4) if active_periods > 0 else None

    equity_curve = pd.DataFrame({
        "open_time": open_time.values, "close": close.values,
        "position": position_applied.values, "signal": signal.values,
        "strategy_return": strategy_return_net.values, "equity": equity.values,
        "buy_hold_equity": buy_hold_equity.values, "drawdown": drawdown.values,
    })

    metrics = {
        "engine": "simple", "initial_capital": initial_capital,
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return_pct": round(total_return * 100, 2),
        "buy_hold_return_pct": round(buy_hold_return * 100, 2),
        "outperformance_pct": round((total_return - buy_hold_return) * 100, 2),
        "max_drawdown_pct": round(float(max_drawdown) * 100, 2),
        "total_trades": total_trades, "win_rate": win_rate,
        "sharpe_like_ratio": round(sharpe_like, 4), "periods": int(len(final_signal)),
    }

    return {"metrics": metrics, "equity_curve": equity_curve}


def _simulate_trades(df: pd.DataFrame, signal: pd.Series, prediction_proba: pd.Series,
                      initial_capital: float, fee_rate: float, slippage_rate: float,
                      spread_rate: float, risk_per_trade: float, sl_atr_mult: float,
                      tp_atr_mult: float, position_mode: str) -> dict:
    """
    هسته‌ی شبیه‌سازی trade-level. position_mode:
      "compounding" - حجم هر معامله بر پایه‌ی سرمایه‌ی فعلی
      "fixed"       - حجم هر معامله همیشه بر پایه‌ی سرمایه‌ی اولیه (بدون ترکیب مضربی)
    """
    n = len(df)
    has_atr = "atr" in df.columns
    atr = df["atr"] if has_atr else pd.Series(np.nan, index=df.index)

    half_cost_rate = slippage_rate + spread_rate / 2

    cash = initial_capital
    units = 0.0
    sl_price = tp_price = None
    open_trade = None

    equity_list, position_list, trade_pnl_list = [], [], []
    trade_ledger = []

    for i in range(n):
        current_close = df.loc[i, "close"]
        trade_pnl_this_row = 0.0

        if units > 0:
            low_i, high_i = df.loc[i, "low"], df.loc[i, "high"]
            exit_price_raw = None
            exit_reason = None

            if sl_price is not None and low_i <= sl_price:
                exit_price_raw, exit_reason = sl_price, "stop_loss"
            elif tp_price is not None and high_i >= tp_price:
                exit_price_raw, exit_reason = tp_price, "take_profit"
            elif i > 0 and signal.iloc[i - 1] == "SELL":
                exit_price_raw, exit_reason = df.loc[i, "open"], "signal_sell"

            if exit_price_raw is not None:
                exit_price_exec = exit_price_raw * (1 - half_cost_rate)
                proceeds = units * exit_price_exec
                exit_fee = proceeds * fee_rate
                cash += proceeds - exit_fee

                gross_pnl = units * (exit_price_exec - open_trade["entry_price_exec"])
                net_pnl = gross_pnl - open_trade["entry_fee"] - exit_fee
                trade_pnl_this_row = net_pnl

                trade_ledger.append({
                    "entry_time": open_trade["entry_time"], "exit_time": int(df.loc[i, "open_time"]),
                    "entry_price": round(open_trade["entry_price_exec"], 4),
                    "exit_price": round(exit_price_exec, 4), "units": round(units, 6),
                    "reason": exit_reason, "fees_paid": round(open_trade["entry_fee"] + exit_fee, 4),
                    "slippage_spread_cost": round(
                        units * open_trade["entry_price_raw"] * half_cost_rate +
                        units * exit_price_raw * half_cost_rate, 4),
                    "net_pnl": round(net_pnl, 4),
                })

                units = 0.0
                sl_price = tp_price = None
                open_trade = None

        if units == 0 and i > 0 and signal.iloc[i - 1] == "BUY":
            entry_price_raw = df.loc[i, "open"]
            entry_price_exec = entry_price_raw * (1 + half_cost_rate)

            atr_i = atr.iloc[i - 1] if has_atr else np.nan
            if pd.notna(atr_i) and atr_i > 0 and entry_price_exec > 0:
                sl_distance = atr_i * sl_atr_mult
                tp_distance = atr_i * tp_atr_mult

                risk_base = initial_capital if position_mode == "fixed" else cash
                risk_amount = risk_base * risk_per_trade
                units_by_risk = risk_amount / sl_distance
                units_by_cash = cash / (entry_price_exec * (1 + fee_rate))
                units_candidate = min(units_by_risk, units_by_cash)

                if units_candidate > 0:
                    cost = units_candidate * entry_price_exec
                    entry_fee = cost * fee_rate
                    cash -= (cost + entry_fee)
                    units = units_candidate
                    sl_price = entry_price_raw - sl_distance
                    tp_price = entry_price_raw + tp_distance
                    open_trade = {
                        "entry_time": int(df.loc[i, "open_time"]),
                        "entry_price_raw": entry_price_raw,
                        "entry_price_exec": entry_price_exec,
                        "entry_fee": entry_fee,
                    }

        position_value = units * current_close
        equity_list.append(cash + position_value)
        position_list.append(1.0 if units > 0 else 0.0)
        trade_pnl_list.append(trade_pnl_this_row)

    return {
        "equity": equity_list, "position": position_list,
        "trade_pnl": trade_pnl_list, "trade_ledger": trade_ledger,
    }


def _compute_full_metrics(df: pd.DataFrame, equity: pd.Series, trade_ledger: list,
                           initial_capital: float) -> dict:
    """
    متریک‌های کامل: Total Return, CAGR, Win Rate (بر پایه PnL واقعی), Profit Factor,
    Expectancy, Max Drawdown, Sharpe, Sortino, Calmar, Average Win/Loss، تعداد معاملات،
    کل کارمزد، کل هزینه‌ی لغزش/اسپرد.
    """
    total_return = float(equity.iloc[-1] / initial_capital - 1)

    elapsed_ms = int(df["open_time"].iloc[-1] - df["open_time"].iloc[0]) or 1
    years = elapsed_ms / MS_PER_YEAR

    # محافظت در برابر overflow: بازه‌ی خیلی کوچیک/غیرواقعی → CAGR بی‌معنیه، محاسبه نکن
    MIN_YEARS_FOR_CAGR = 1 / 365
    cagr = None
    if years >= MIN_YEARS_FOR_CAGR and equity.iloc[-1] > 0:
        try:
            cagr = float(equity.iloc[-1] / initial_capital) ** (1 / years) - 1
            if not np.isfinite(cagr):
                cagr = None
        except OverflowError:
            cagr = None

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max.replace(0, np.nan)
    max_drawdown = float(drawdown.min()) if not drawdown.isna().all() else 0.0

    median_gap_ms = df["open_time"].diff().median()
    periods_per_year = (MS_PER_YEAR / median_gap_ms) if median_gap_ms and median_gap_ms > 0 else 252

    period_returns = equity.pct_change().fillna(0.0)
    ret_std = period_returns.std()
    sharpe = float((period_returns.mean() / ret_std) * np.sqrt(periods_per_year)) if ret_std and ret_std > 0 else 0.0

    downside_returns = period_returns[period_returns < 0]
    downside_std = downside_returns.std()
    sortino = float((period_returns.mean() / downside_std) * np.sqrt(periods_per_year)) if downside_std and downside_std > 0 else 0.0

    calmar = float(cagr / abs(max_drawdown)) if cagr is not None and max_drawdown != 0 else None

    pnls = [t["net_pnl"] for t in trade_ledger]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n_trades = len(pnls)

    win_rate = round(len(wins) / n_trades, 4) if n_trades > 0 else None
    avg_win = round(float(np.mean(wins)), 4) if wins else 0.0
    avg_loss = round(float(np.mean(losses)), 4) if losses else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = round(gross_profit / gross_loss, 4) if gross_loss > 0 else (None if gross_profit == 0 else float("inf"))
    expectancy = round(float(np.mean(pnls)), 4) if pnls else None

    total_fees = round(sum(t["fees_paid"] for t in trade_ledger), 2)
    total_slippage_spread = round(sum(t["slippage_spread_cost"] for t in trade_ledger), 2)

    return {
        "initial_capital": initial_capital,
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2) if cagr is not None else None,
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4) if calmar is not None else None,
        "total_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy_per_trade": expectancy,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "total_fees_paid": total_fees,
        "total_slippage_spread_cost": total_slippage_spread,
    }


def run_backtest_advanced(df: pd.DataFrame, final_signal: pd.Series,
                           initial_capital: float = BACKTEST_INITIAL_CAPITAL,
                           fee_rate: float = BACKTEST_FEE_RATE,
                           slippage_rate: float = BACKTEST_SLIPPAGE_RATE,
                           spread_rate: float = BACKTEST_SPREAD_RATE,
                           risk_per_trade: float = BACKTEST_RISK_PER_TRADE,
                           sl_atr_mult: float = BACKTEST_SL_ATR_MULTIPLIER,
                           tp_atr_mult: float = BACKTEST_TP_ATR_MULTIPLIER,
                           prediction_proba: pd.Series = None,
                           position_mode: str = "compounding") -> dict:
    """موتور پیشرفته - یک حالت. برای مقایسه‌ی هر دو حالت از run_backtest_dual استفاده کن."""
    df = sanitize_ohlc_for_backtest(df).reset_index(drop=True)
    signal = final_signal.reset_index(drop=True)
    proba = prediction_proba.reset_index(drop=True) if prediction_proba is not None else pd.Series(np.nan, index=df.index)

    sim = _simulate_trades(df, signal, proba, initial_capital, fee_rate, slippage_rate,
                            spread_rate, risk_per_trade, sl_atr_mult, tp_atr_mult, position_mode)

    equity = pd.Series(sim["equity"], index=df.index)
    market_return = df["close"].pct_change().fillna(0.0)
    buy_hold_equity = initial_capital * (1 + market_return).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max.replace(0, np.nan)

    metrics = _compute_full_metrics(df, equity, sim["trade_ledger"], initial_capital)
    metrics["engine"] = f"advanced ({position_mode})"
    metrics["buy_hold_return_pct"] = round(float(buy_hold_equity.iloc[-1] / initial_capital - 1) * 100, 2)
    metrics["outperformance_pct"] = round(metrics["total_return_pct"] - metrics["buy_hold_return_pct"], 2)
    metrics["risk_per_trade_pct"] = round(risk_per_trade * 100, 2)
    metrics["sl_atr_multiplier"] = sl_atr_mult
    metrics["tp_atr_multiplier"] = tp_atr_mult
    metrics["periods"] = int(len(df))

    equity_curve = pd.DataFrame({
        "open_time": df["open_time"].values, "close": df["close"].values,
        "position": sim["position"], "signal": signal.values,
        "prediction_proba": proba.values, "equity": equity.values,
        "buy_hold_equity": buy_hold_equity.values, "drawdown": drawdown.values,
        "trade_pnl": sim["trade_pnl"],
    })

    return {"metrics": metrics, "equity_curve": equity_curve, "trade_log": sim["trade_ledger"]}


def run_backtest_dual(df: pd.DataFrame, final_signal: pd.Series, prediction_proba: pd.Series = None, **kwargs) -> dict:
    """هر دو حالت compounding و fixed رو اجرا می‌کنه تا معلوم بشه سود از کجا میاد."""
    compounding = run_backtest_advanced(df, final_signal, prediction_proba=prediction_proba,
                                         position_mode="compounding", **kwargs)
    fixed = run_backtest_advanced(df, final_signal, prediction_proba=prediction_proba,
                                   position_mode="fixed", **kwargs)
    return {"compounding": compounding, "fixed": fixed}
