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

# ==========================================
# ۱. تنظیمات پیشرفته ظاهری (Mobile First UI)
# ==========================================
st.set_page_config(page_title="AI Trader", layout="centered") # برای موبایل centered بهتر است

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@100;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Vazirmatn', sans-serif;
        direction: rtl;
        text-align: right;
    }
    
    /* استایل کارت‌های شاخص */
    .stMetric {
        background: #1e222d;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #31353f;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* استایل دکمه اصلی */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 55px;
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white;
        font-weight: bold;
        font-size: 18px;
        border: none;
        margin-top: 10px;
    }
    
    /* استایل باکس سیگنال */
    .signal-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
        font-weight: bold;
        font-size: 20px;
    }
    
    .long-bg { background-color: rgba(0, 184, 148, 0.2); border: 2px solid #00b894; color: #00b894; }
    .short-bg { background-color: rgba(214, 48, 49, 0.2); border: 2px solid #d63031; color: #d63031; }
    .neutral-bg { background-color: rgba(178, 190, 195, 0.1); border: 2px solid #636e72; color: #636e72; }
    </style>
    """, unsafe_allow_html=True)

TOKEN = "8548739067:AAGuvMHgB-LxOoyQIrHWzs6ytTfOehfIrco"
CHAT_ID = "163583693"
CRYPTOS = {"BTC": "BTC/USDT", "ETH": "ETH/USDT", "XRP": "XRP/USDT", "SOL": "SOL/USDT"}

# ==========================================
# ۲. توابع (بدون تغییر در منطق)
# ==========================================

def get_data(coin_key, interval="1h"):
    symbol = CRYPTOS.get(coin_key, "BTC/USDT")
    exchange = ccxt.kucoin({'enableRateLimit': True})
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=200)
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
        df["rsi"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        df["macd"] = ta.trend.MACD(df["price"]).macd_diff()
        df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["price"]).adx()
        df["ema"] = ta.trend.EMAIndicator(df["price"], 20).ema_indicator()
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["price"]).average_true_range()
        return df.dropna()
    except: return None

def train_xgb(df):
    try:
        features = ["rsi", "macd", "ema", "atr", "adx"]
        X, y = df[features].copy(), (df["price"].shift(-1) > df["price"]).astype(int)
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
# ۳. رابط کاربری مخصوص موبایل
# ==========================================

st.title("🤖 دستیار ترید AI")

# تنظیمات در سایدبار (برای خلوت شدن صفحه اصلی)
with st.sidebar:
    st.header("⚙️ تنظیمات حساب")
    capital = st.number_input("سرمایه کل ($)", value=1000)
    risk_pct = st.slider("درصد ریسک", 1.0, 5.0, 2.0)
    st.markdown("---")
    st.write("نسخه اپلیکیشن: 14.2")

# بخش انتخاب سریع
coin_choice = st.selectbox("انتخاب ارز دیجیتال", list(CRYPTOS.keys()))
tf_choice = st.selectbox("تایم‌فریم تحلیل", ["15m", "1h", "4h", "1d"])

if st.button("🚀 شروع تحلیل هوشمند"):
    with st.spinner('در حال اسکن بازار...'):
        raw_df = get_data(coin_choice, tf_choice)
        df = add_indicators(raw_df)
        
        if df is not None:
            xgb_p = train_xgb(df)
            lstm_p = train_lstm(df)
            
            price = df['price'].iloc[-1]
            ensemble = (xgb_p * 0.5) + (lstm_p * 0.5)
            adx = df['adx'].iloc[-1]
            atr = df['atr'].iloc[-1]
            
            # تعیین استایل سیگنال
            if ensemble > 70 and adx > 18:
                sig_class, sig_text = "long-bg", "سیگنال خرید (LONG) 🟩"
            elif ensemble < 30 and adx > 18:
                sig_class, sig_text = "short-bg", "سیگنال فروش (SHORT) 🟥"
            else:
                sig_class, sig_text = "neutral-bg", "وضعیت خنثی / صبر ⬜"

            # نمایش قیمت بزرگ در بالا
            st.metric("قیمت لحظه‌ای", f"${price:,.4f}")
            
            # نمایش باکس سیگنال
            st.markdown(f'<div class="signal-box {sig_class}">{sig_text}</div>', unsafe_allow_html=True)
            
            # کارت‌های جزئیات
            col_a, col_b = st.columns(2)
            col_a.metric("اطمینان هوش مصنوعی", f"{ensemble:.1f}%")
            col_b.metric("قدرت روند (ADX)", f"{adx:.1f}")
            
            # بخش مدیریت معامله
            st.markdown("### 🎯 جزئیات معامله")
            sl = price - (2.5 * atr) if ensemble > 50 else price + (2.5 * atr)
            tp = price + (1.5 * abs(price - sl)) if ensemble > 50 else price - (1.5 * abs(price - sl))
            
            risk_amt = capital * (risk_pct / 100)
            pos_size = risk_amt / abs(price - sl) * price
            
            st.info(f"**حد سود (TP):** {tp:,.4f}")
            st.error(f"**حد ضرر (SL):** {sl:,.4f}")
            st.success(f"**حجم ورود پیشنهادی:** ${pos_size:,.2f}")
            
        else:
            st.error("خطا در دریافت دیتا! لطفاً دوباره تلاش کنید.")

gc.collect()
