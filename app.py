import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import ta
import gc
import ccxt
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# =========================
# ۱. تنظیمات و ظاهر (UI)
# =========================
st.set_page_config(page_title="AI-CRYPTO ELITE v14.0", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn&display=swap');
    html, body, [class*="css"] { font-family: 'Vazirmatn', sans-serif; direction: rtl; text-align: right; }
    .stMetric { background: #1a1c23; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

TOKEN = "8548739067:AAGuvMHgB-LxOoyQIrHWzs6ytTfOehfIrco"
CHAT_ID = "163583693"
# فرمت ارزها برای کوکوین
CRYPTOS = {"bitcoin": "BTC/USDT", "ethereum": "ETH/USDT", "ripple": "XRP/USDT", "solana": "SOL/USDT"}

# =========================
# ۲. توابع ارتباطی و داده
# =========================

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=5)
    except: pass

def get_data(coin, interval="1h"):
    symbol = CRYPTOS.get(coin, "BTC/USDT")
    exchange = ccxt.kucoin({'enableRateLimit': True})
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=300)
        if len(ohlcv) < 100: return None
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df["price"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df.set_index('ts', inplace=True)
        return df
    except: return None

def add_indicators(df):
    try:
        if df is None or len(df) < 50: return None
        df["rsi"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        df["macd"] = ta.trend.MACD(df["price"]).macd_diff()
        df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["price"]).adx()
        df["ema"] = ta.trend.EMAIndicator(df["price"], 20).ema_indicator()
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["price"]).average_true_range()
        return df.dropna()
    except: return None

# =========================
# ۳. مدل‌های هوش مصنوعی
# =========================

def train_xgb(df):
    try:
        features = ["rsi", "macd", "ema", "atr", "adx"]
        X = df[features].copy()
        y = (df["price"].shift(-1) > df["price"]).astype(int)
        X, y = X[:-1], y[:-1]
        model = XGBClassifier(n_estimators=30, max_depth=3, verbosity=0)
        model.fit(X, y)
        return model.predict_proba(X.iloc[-1:])[0][1] * 100
    except: return 50

def train_lstm(df):
    try:
        K.clear_session()
        data = df[["price"]].values
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)
        X_train = np.array([scaled[-51:-1]])
        model = Sequential([LSTM(16, input_shape=(50, 1)), Dense(1)])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, np.array([scaled[-1]]), epochs=1, verbose=0)
        pred = model.predict(X_train, verbose=0)[0][0]
        return 100 if pred > scaled[-1][0] else 0
    except: return 50

# =========================
# ۴. بدنه اصلی اپلیکیشن
# =========================

st.title("💎 پنل ترید هوشمند Elite AI")

with st.sidebar:
    st.header("⚙️ مدیریت سرمایه")
    capital = st.number_input("سرمایه کل (USD)", value=1000)
    risk_pct = st.slider("درصد ریسک هر معامله", 1.0, 5.0, 2.0)
    st.info("داده‌ها از صرافی Kucoin دریافت می‌شوند.")

c1, c2 = st.columns(2)
with c1:
    coin_select = st.selectbox("انتخاب ارز:", list(CRYPTOS.keys()))
with c2:
    tf_select = st.selectbox("تایم‌فریم:", ["15m", "1h", "4h", "1d"])

if st.button("🚀 تحلیل و صدور سیگنال"):
    with st.spinner('در حال پردازش داده‌ها و آموزش مدل‌ها...'):
        raw_df = get_data(coin_select, tf_select)
        df = add_indicators(raw_df)
        
        if df is not None and not df.empty:
            # آموزش مدل‌ها
            xgb_p = train_xgb(df)
            lstm_p = train_lstm(df)
            
            # محاسبات قیمت و روند
            price = df['price'].iloc[-1]
            ensemble = (xgb_p * 0.5) + (lstm_p * 0.5)
            adx = df['adx'].iloc[-1]
            atr = df['atr'].iloc[-1]
            
            # منطق سیگنال (دقیقاً مشابه نسخه ۱۳ شما)
            signal_text = "خنثی / صبر ⬜"
            if ensemble > 70 and adx > 18: signal_text = "خرید (LONG) 🟩"
            elif ensemble < 30 and adx > 18: signal_text = "فروش (SHORT) 🟥"
            
            # مدیریت ریسک
            sl = price - (2.5 * atr) if ensemble > 50 else price + (2.5 * atr)
            tp = price + (1.5 * abs(price - sl)) if ensemble > 50 else price - (1.5 * abs(price - sl))
            
            # نمایش خروجی در اپلیکیشن
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("قیمت لحظه‌ای", f"${price:.4f}")
            m2.metric("اطمینان هوش مصنوعی", f"{ensemble:.1f}%")
            m3.metric("سیگنال نهایی", signal_text)
            
            res_c1, res_c2 = st.columns(2)
            with res_c1:
                st.success(f"🎯 تارگت پیشنهادی: {tp:.4f}")
                st.error(f"🛡️ حد ضرر (SL): {sl:.4f}")
            with res_c2:
                risk_amt = capital * (risk_pct / 100)
                pos_size = risk_amt / abs(price - sl) * price
                st.info(f"📏 حجم پوزیشن: ${pos_size:.2f}")

            # ارسال پیام به تلگرام
            tg_msg = f"🚀 **سیگنال جدید {coin_select.upper()}**\n"
            tg_msg += f"وضعیت: {signal_text}\n"
            tg_msg += f"قیمت: {price:.4f}\n"
            tg_msg += f"قدرت پیش‌بینی: {ensemble:.1f}%\n"
            tg_msg += f"تارگت: {tp:.4f} | استاپ: {sl:.4f}"
            send_telegram(tg_msg)
            
        else:
            st.error("❌ خطا در دریافت یا پردازش دیتا. لطفاً تایم‌فریم را تغییر دهید.")

gc.collect()
