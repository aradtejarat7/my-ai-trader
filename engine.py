import ta
import numpy as np
from xgboost import XGBClassifier

def get_market_regime(df):
    adx = df['adx'].iloc[-1]
    if adx > 25: return "Trend 💪 (رونددار)"
    if adx < 20: return "Range 💤 (بدون روند)"
    return "Stable ⚖️ (معمولی)"

def calculate_management(price, sl, capital, risk_pct):
    risk_amt = capital * (risk_pct / 100)
    price_diff = abs(price - sl)
    if price_diff == 0: return 0, 0
    qty = risk_amt / price_diff
    pos_size = qty * price
    return round(pos_size, 2), round(qty, 4)

def get_targets(price, sl, signal_type):
    diff = abs(price - sl)
    if signal_type == "LONG":
        return price + (diff * 0.8), price + (diff * 1.5), price + (diff * 2.5)
    else:
        return price - (diff * 0.8), price - (diff * 1.5), price - (diff * 2.5)

def get_ml_probs(df):
    # در اینجا منطق XGB و LSTM که قبلاً داشتیم اجرا می‌شود
    # برای مثال خروجی فرضی:
    return 48, 0 # XGB, LSTM
