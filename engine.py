import ta
import numpy as np
from xgboost import XGBClassifier

def add_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["price"]).rsi()
    df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["price"]).adx()
    df["ema"] = ta.trend.EMAIndicator(df["price"], 20).ema_indicator()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["price"]).average_true_range()
    return df.dropna()

def get_ml_prediction(df):
    features = ["rsi", "adx", "ema"]
    X = df[features].copy()
    y = (df["price"].shift(-1) > df["price"]).astype(int)
    model = XGBClassifier(n_estimators=30, max_depth=3)
    model.fit(X[:-1], y[:-1])
    prob = model.predict_proba(X.iloc[-1:])[0][1]
    return int(prob * 100)
