import streamlit as st
from connector import get_data, CRYPTOS
from engine import add_indicators, calculate_trade_details
from styles import apply_styles

apply_styles()

st.title("💎 AI-CRYPTO ELITE v13.0")

with st.sidebar:
    capital = st.number_input("سرمایه ($)", value=1000)
    risk = st.slider("ریسک (%)", 1.0, 5.0, 2.0)

# تعریف متغیرها با Selectbox (جلوگیری از NameError)
coin = st.selectbox("ارز:", list(CRYPTOS.keys()))
tf = st.selectbox("تایم‌فریم:", ["15m", "1h", "4h", "1d"])

if st.button("🚀 صدور سیگنال هوشمند"):
    with st.spinner("در حال تحلیل..."):
        df_raw = get_data(coin, tf)
        if df_raw is not None:
            df = add_indicators(df_raw)
            res = calculate_trade_details(df, capital, risk)
            
            # نمایش دقیق مشابه فرمتی که خواستید
            st.markdown(f"""
            <div class="signal-card">
                <div style="text-align:center; font-weight:bold; font-size:18px;">💎 AI-CRYPTO ELITE v13.0</div>
                💰 ارز: {coin} | ⏱️ {tf}<br>
                ───────────────────<br>
                🎯 <b>سیگنال: {res['sig']}</b><br>
                📊 وضعیت: {"خرید 🟩" if "LONG" in res['sig'] else "فروش 🟥"}<br>
                📈 اعتماد مدل: {res['adx']:.1f}%<br><br>
                💵 قیمت فعلی: ${res['price']:.4f}<br>
                🛡️ حد ضرر: ${res['sl']:.4f}<br>
                🎯 تارگت ۱: ${res['tp1']:.4f}<br>
                🎯 تارگت ۲: ${res['tp2']:.4f}<br>
                🎯 تارگت ۳: ${res['tp3']:.4f}<br><br>
                💰 <b>مدیریت سرمایه:</b><br>
                📏 حجم پوزیشن: ${res['pos_size']:.2f}<br>
                📈 تعداد واحد: {res['qty']:.4f} {coin}<br>
                ───────────────────<br>
                📊 <b>اندیکاتورها:</b><br>
                • ADX: {res['adx']:.1f} | RSI: {res['rsi']:.1f}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("خطا در دریافت دیتا")
