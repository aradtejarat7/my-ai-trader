import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import ta
import gc
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# ==========================================
# ۱. پیکربندی صفحه و استایل
# ==========================================
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
CRYPTOS = {"bitcoin": "BTCUSDT", "ethereum": "ETHUSDT", "ripple": "XRPUSDT", "solana": "SOLUSDT"}

# ==========================================
# ۲. توابع داده و محاسبات
# ==========================================

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=5)
    except: pass

def get_futures_info(symbol):
    try:
        f_url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
        oi_url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
        fund = float(requests.get(f_url, timeout=5).json().get("lastFundingRate", 0)) * 100
        oi = float(requests.get(oi_url, timeout=5).json().get("openInterest", 0))
        return fund, oi
    except: return 0.0, 0.0

import ccxt

def get_data(coin, interval="1h"):
    symbol = CRYPTOS.get(coin, "BTCUSDT")
    # استفاده از CCXT برای پایداری بیشتر در سرورهای ابری
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    for attempt in range(3): # ۳ بار تلاش مجدد در صورت خطا
        try:
            # دریافت داده‌های کندل استیک
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=200)
            if not ohlcv or len(ohlcv) < 100:
                continue
                
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            df["price"] = df["close"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["volume"] = df["volume"].astype(float)
            
            # دریافت اطلاعات فاندینگ و OI (اختیاری و با مدیریت خطا)
            try:
                fund, oi = get_futures_info(symbol)
                df["funding_rate"], df["open_interest"] = fund, oi
            except:
                df["funding_rate"], df["open_interest"] = 0.0, 0.0
                
            return df
        except Exception as e:
            time.sleep(1) # وقفه کوتاه قبل از تلاش مجدد
            continue
            
    return None

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

# ==========================================
# ۳. توابع هوش مصنوعی
# ==========================================

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

# ==========================================
# ۴. بخش اصلی اپلیکیشن (UI)
# ==========================================

st.title("🚀 پنل هوش مصنوعی Elite v14.0")

with st.sidebar:
    st.header("⚙️ تنظیمات")
    capital = st.number_input("سرمایه (USD)", value=1000)
    risk_pct = st.slider("ریسک (%)", 1.0, 5.0, 2.0)

col1, col2 = st.columns(2)
with col1:
    coin_select = st.selectbox("انتخاب ارز:", list(CRYPTOS.keys()))
with col2:
    tf_select = st.selectbox("تایم‌فریم:", ["15m", "1h", "4h", "1d"])

if st.button("🔍 شروع تحلیل هوشمند"):
    with st.spinner('در حال پردازش مدل‌های هوشمند...'):
        raw_df = get_data(coin_select, tf_select)
        df = add_indicators(raw_df)
        
        if df is not None and not df.empty:
            xgb_p = train_xgb(df)
            lstm_p = train_lstm(df)
            
            price = df['price'].iloc[-1]
            ensemble = (xgb_p * 0.5) + (lstm_p * 0.5)
            adx = df['adx'].iloc[-1]
            
            # منطق سیگنال
            signal = "صبر / خنثی ⬜"
            if ensemble > 70 and adx > 18: signal = "خرید (LONG) 🟩"
            elif ensemble < 30 and adx > 18: signal = "فروش (SHORT) 🟥"
            
            # محاسبات مدیریت سرمایه
            atr = df['atr'].iloc[-1]
            sl = price - (2.5 * atr) if ensemble > 50 else price + (2.5 * atr)
            tp = price + (1.5 * abs(price - sl)) if ensemble > 50 else price - (1.5 * abs(price - sl))
            
            # نمایش خروجی
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("قیمت لحظه‌ای", f"${price:.4f}")
            m2.metric("اطمینان مدل", f"{ensemble:.1f}%")
            m3.metric("سیگنال", signal)
            
            res_c1, res_c2 = st.columns(2)
            with res_c1:
                st.success(f"🎯 حد سود: {tp:.4f}")
                st.error(f"🛡️ حد ضرر: {sl:.4f}")
            with res_c2:
                risk_amt = capital * (risk_pct / 100)
                pos_size = risk_amt / abs(price - sl) * price
                st.info(f"📏 حجم پیشنهادی: ${pos_size:.2f}")

            # ارسال تلگرام
            msg = f"🚀 سیگنال جدید {coin_select.upper()}\n💰 قیمت: {price}\n📊 قدرت: {ensemble:.1f}%\n🎯 هدف: {tp:.4f}"
            send_telegram(msg)
        else:
            st.error("❌ خطا در دریافت دیتا. لطفاً دوباره تلاش کنید.")

# پاکسازی خودکار حافظه
gc.collect()
