"""
فاز ۱۰ - REST API (FastAPI) - آخرین سیگنال به‌همراه اعتماد و توضیح
"""

from fastapi import FastAPI, HTTPException

from config import FEATURE_VERSION
from models.predict import load_models, predict_proba_up, load_feature_columns
from models.dataset_prep import build_ml_dataset
from rule_engine.rules import compute_rule_signals
from features.build_features import load_feature_store
from signals.aggregator import aggregate_ml_scores, aggregate_signals
from signals.confidence import compute_confidence
from signals.explainability import get_global_feature_importance, explain_single_signal

app = FastAPI(title="Bitcoin Intelligence Engine API", version="0.1.0")


@app.get("/")
def root():
    return {"status": "ok", "service": "Bitcoin Intelligence Engine", "feature_version": FEATURE_VERSION}


@app.get("/signal/{symbol}/{interval}")
def get_latest_signal(symbol: str, interval: str):
    models = load_models(symbol, interval)
    if not models:
        raise HTTPException(status_code=404, detail=f"مدلی برای {symbol}/{interval} train نشده.")

    df = load_feature_store(symbol, interval, FEATURE_VERSION)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"داده‌ای در feature_store برای {symbol}/{interval} نیست.")

    try:
        X_train, X_test, y_train, y_test, feature_cols, full_df = build_ml_dataset(symbol, interval)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    last_idx = X_test.index[-1]
    last_X = X_test.loc[[last_idx]]

    proba_df = predict_proba_up(models, last_X)
    ml_score = aggregate_ml_scores(proba_df)

    rule_signals_df = compute_rule_signals(df)
    rule_score = rule_signals_df.loc[[last_idx], "rule_engine_score"]

    agg = aggregate_signals(ml_score, rule_score)
    conf = compute_confidence(full_df.loc[[last_idx]], proba_df, agg["final_signal"], y_test.loc[[last_idx]])

    importance = get_global_feature_importance(models, feature_cols)
    explanation = explain_single_signal(full_df.loc[last_idx], importance, rule_signals_df.loc[last_idx])

    return {
        "symbol": symbol, "interval": interval,
        "open_time": int(full_df.loc[last_idx, "open_time"]),
        "signal": agg.loc[last_idx, "final_signal"],
        "final_score": round(float(agg.loc[last_idx, "final_score"]), 4),
        "ml_score": round(float(agg.loc[last_idx, "ml_score"]), 4),
        "rule_score": round(float(agg.loc[last_idx, "rule_score"]), 4),
        "confidence": round(float(conf.loc[last_idx, "confidence"]), 4),
        "explanation": explanation,
        "disclaimer": "این یک تحلیل خودکار است، نه توصیه مالی یا سیگنال معاملاتی تضمینی.",
    }
