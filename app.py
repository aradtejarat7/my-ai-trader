import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import ta
import gc
import asyncio
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# ==========================================
# ۱. تنظیمات اولیه و استایل (UI CONFIG)
# ==========================================
st.set_page_config(page_title="AI-CRYPTO ELITE v14.0", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn&display=swap');
    html, body, [class*="css"] { font-family: 'Vazirmatn', sans-serif; direction: rtl; text-align: right; }
    .reportview-container { background: #0e1117; }
    .stMetric { background: #1a1c23; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .stAlert { direction: rtl; }
    </style>
    """, unsafe_allow_html=True)

# توکن و تنظیمات کاربر (از کدهای قبلی شما)
TOKEN = "8548739067:AAGuvMHgB-LxOoyQIrHWzs6ytTfOehfIrco"
CHAT_ID = "163583693"
CRYPTOS = {"bitcoin": "BTC", "ethereum": "ETH", "ripple": "XRP", "solana": "SOL"}

# ==========================================
# ۲. توابع اصلی (بدون حذف هیچ منطقی)
# ==========================================

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=10)
    except: pass

def get_futures_info(symbol):
    try:
        fund_url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
        fund_data = requests.get(fund_url, timeout=10).json()
        funding_rate = float(fund_data.get("lastFundingRate", 0)) * 100
        oi_url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
        oi_data = requests.get(oi_url, timeout=10).json()
        open_interest = float(oi_data.get("openInterest", 0))
        return funding_rate, open_interest
    except: return 0.0, 0.0

def get_data(coin, interval="1h", candles=1000):
    symbol_map = {"bitcoin": "BTCUSDT", "ethereum": "ETHUSDT", "ripple": "XRPUSDT", "solana": "SOLUSDT"}
    symbol = symbol_map.get(coin, "BTCUSDT")
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={candles}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200: return None
        data = r.json()
        
        # بررسی اینکه آیا حداقل ۱۰۰ کندل برای محاسبات موجود است
        if len(data) < 100: return None
        
        df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume", "ct", "qav", "nt", "tb", "tq", "i"])
        df["price"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)
        
        funding, oi = get_futures_info(symbol)
        df["funding_rate"], df["open_interest"] = funding, oi
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)
        return df
    except Exception as e:
        return None

def add_indicators(df):
    try:
        # اگر تعداد ردیف‌ها کمتر از ۵۰ باشد، محاسبات ADX خطا می‌دهد
        if df is None or len(df) < 50:
            return None
            
        df["rsi"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        df["macd"] = ta.trend.MACD(df["price"]).macd_diff()
        
        # ADX با پنجره ۱۴ تایی
        adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["price"], window=14)
        df["adx"] = adx_ind.adx()
        
        df["ema"] = ta.trend.EMAIndicator(df["price"], 20).ema_indicator()
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["price"]).average_true_range()
        df["vol_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()
        df["oi_change"] = df["open_interest"].pct_change() * 100
        
        return df.dropna()
    except Exception as e:
        return None


# --- توابع ML با بهبود پایداری ---
def train_xgb(df):
    try:
        features = ["rsi", "macd", "ema", "atr", "adx", "vol_ratio"]
        X, y = df[features].copy(), (df["price"].shift(-1) > df["price"]).astype(int)
        X, y = X[:-1], y[:-1]
        model = XGBClassifier(n_estimators=50, max_depth=3, verbosity=0)
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

def get_btc_bias():
    try:
        df_btc = get_data("bitcoin", interval="1h")
        # بررسی اینکه آیا دیتا دریافت شده و خالی نیست
        if df_btc is None or df_btc.empty or len(df_btc) < 50:
            return "در حال بارگذاری... ⏳", "UNKNOWN"
            
        last_p = df_btc['price'].iloc[-1]
        ema = df_btc['price'].ewm(span=50).mean().iloc[-1]
        
        if last_p > ema:
            return "صعودی 🟢", "BULLISH"
        else:
            return "نزولی 🔴", "BEARISH"
    except Exception as e:
        # در صورت هرگونه خطا، برنامه کرش نمی‌کند
        return "عدم دسترسی به دیتا ⚪", "UNKNOWN"

# ==========================================
# ۳. رابط کاربری اپلیکیشن (STREAMLIT UI)
# ==========================================

st.title("🚀 سیستم معاملاتی Elite AI v14.0")

with st.sidebar:
    st.header("⚙️ تنظیمات مدیریت سرمایه")
    my_capital = st.number_input("کل سرمایه (دلار)", value=1000)
    risk_per_trade = st.slider("ریسک در هر معامله (%)", 0.5, 5.0, 2.0)
    st.divider()
    st.write("وضعیت بازار:")
    bias_text, bias_val = get_btc_bias()
    st.subheader(bias_text)

# انتخاب ارز و تایم‌فریم
col1, col2 = st.columns(2)
with col1:
    coin_choice = st.selectbox("ارز مورد نظر:", list(CRYPTOS.keys()))
with col2:
    tf_choice = st.selectbox("تایم‌فریم:", ["15m", "1h", "4h", "1d"])

# این بخش دقیقاً بعد از انتخاب ارز و تایم‌فریم قرار می‌گیرد

if st.button("🔍 اجرای تحلیل عمیق و صدور سیگنال"):
    with st.spinner('در حال دریافت دیتا و آموزش هوش مصنوعی...'):
        # ۱. دریافت دیتا از صرافی
        df = get_data(coin_choice, tf_choice)
        
        if df is not None:
            # ۲. اضافه کردن اندیکاتورها با بررسی سلامت دیتا
            df_final = add_indicators(df)
            
            if df_final is not None and not df_final.empty:
                # ۳. اجرای مدل‌های هوش مصنوعی (فقط اگر دیتا سالم باشد)
                xgb_p = train_xgb(df_final)
                lstm_p = train_lstm(df_final)
                
                # ادامه محاسبات (قیمت، حد ضرر، تارگت و ...)
                price = df_final['price'].iloc[-1]
                ema_val = df_final['ema'].iloc[-1]
                trend_score = 100 if price > ema_val else 0
                ensemble = (xgb_p * 0.45) + (lstm_p * 0.40) + (trend_score * 0.15)
                
                # --- نمایش نتایج در داشبورد ---
                st.divider()
                st.balloons() # یک افکت گرافیکی برای جذابیت موقع سیگنال
                
                m1, m2, m3 = st.columns(3)
                m1.metric("قیمت لحظه‌ای", f"${price:.4f}")
                m2.metric("اطمینان هوش مصنوعی", f"{ensemble:.1f}%")
                
                # تعیین سیگنال
                adx = df_final['adx'].iloc[-1]
                signal = "NEUTRAL ⚪"
                if ensemble > 70 and adx > 18: signal = "STRONG_LONG 🟩"
                elif ensemble < 30 and adx > 18: signal = "STRONG_SHORT 🟥"
                
                m3.metric("سیگنال نهایی", signal)
                
                # نمایش جزئیات مدیریت سرمایه (تارگت و استاپ)
                # ... (بقیه کدهای نمایش که در پیام‌های قبلی بود)
                
            else:
                # نمایش خطا به کاربر اگر دیتا برای اندیکاتورها کم بود
                st.error("❌ دیتای کافی برای این تایم‌فریم یافت نشد. لطفاً تایم‌فریم بزرگتری را انتخاب کنید یا چند لحظه دیگر امتحان کنید.")
        else:
            st.error("❌ خطا در اتصال به بایننس. لطفاً وضعیت اینترنت سرور یا نام ارز را چک کنید.")

            
            
            # محاسبه امتیاز نهایی
            price = df['price'].iloc[-1]
            ema_val = df['ema'].iloc[-1]
            trend_score = 100 if price > ema_val else 0
            ensemble = (xgb_p * 0.45) + (lstm_p * 0.40) + (trend_score * 0.15)
            
            # رژیم بازار
            adx = df['adx'].iloc[-1]
            regime = "Trend 💪" if adx > 25 else "Range 💤" if adx < 20 else "Stable ⚖️"
            
            # سیگنال دهی
            signal = "NEUTRAL"
            if ensemble > 70 and adx > 18: signal = "STRONG_LONG 🟩"
            elif ensemble < 30 and adx > 18: signal = "STRONG_SHORT 🟥"
            
            # مدیریت ریسک (محاسبه حد ضرر و حجم)
            atr = df['atr'].iloc[-1]
            sl = price - (2.5 * atr) if "LONG" in signal else price + (2.5 * atr)
            tp1 = price + (1.2 * atr) if "LONG" in signal else price - (1.2 * atr)
            
            risk_amt = my_capital * (risk_per_trade / 100)
            qty = risk_amt / abs(price - sl)
            pos_size = qty * price

            # نمایش نتایج در اپلیکیشن
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("قیمت فعلی", f"${price:.4f}")
            m2.metric("اطمینان مدل", f"{ensemble:.1f}%")
            m3.metric("رژیم بازار", regime)

            st.success(f"### نتیجه تحلیل: {signal}")
            
            res1, res2 = st.columns(2)
            with res1:
                st.write(f"🛡️ **حد ضرر:** {sl:.4f}")
                st.write(f"🎯 **تارگت اصلی:** {tp1:.4f}")
            with res2:
                st.write(f"📏 **حجم پوزیشن:** ${pos_size:.2f}")
                st.write(f"📈 **تعداد واحد:** {qty:.4f}")

            # ارسال به تلگرام (همان تابعی که داشتید)
            final_msg = f"💎 AI-CRYPTO ELITE\nارز: {coin_choice.upper()}\nسیگنال: {signal}\nقیمت: {price}\nتارگت: {tp1:.4f}\nاستاپ: {sl:.4f}"
            send_telegram(final_msg)
            
            st.link_button("👁️ مشاهده در TradingView", f"https://www.tradingview.com/chart/?symbol=BINANCE:{CRYPTOS[coin_choice]}USDT")
        else:
            st.error("خطا در دریافت داده!")
