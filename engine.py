import ta
import numpy as np

def add_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["price"]).rsi()
    df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["price"]).adx()
    df["ema"] = ta.trend.EMAIndicator(df["price"], 20).ema_indicator()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["price"]).average_true_range()
    return df.dropna()

def calculate_trade_details(df, capital, risk_pct):
    price = df['price'].iloc[-1]
    atr = df['atr'].iloc[-1]
    rsi = df['rsi'].iloc[-1]
    adx = df['adx'].iloc[-1]
    
    # منطق جهت معامله
    is_long = rsi > 50 and price > df['ema'].iloc[-1]
    sig_type = "STRONG_LONG" if is_long else "STRONG_SHORT"
    
    # محاسبات SL و TP
    sl_dist = 2.5 * atr
    sl = price - sl_dist if is_long else price + sl_dist
    tp1 = price + (sl_dist * 0.8) if is_long else price - (sl_dist * 0.8)
    tp2 = price + (sl_dist * 1.5) if is_long else price - (sl_dist * 1.5)
    tp3 = price + (sl_dist * 2.5) if is_long else price - (sl_dist * 2.5)
    
    # مدیریت سرمایه
    risk_amt = capital * (risk_pct / 100)
    qty = risk_amt / abs(price - sl)
    pos_size = qty * price
    
    return {
        "price": price, "sig": sig_type, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "pos_size": pos_size, "qty": qty, "adx": adx, "rsi": rsi
    }
