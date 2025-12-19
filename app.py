import streamlit as st
from connector import get_data, CRYPTOS
from engine import add_indicators, get_ml_prediction
from styles import apply_mobile_styles
from datetime import datetime
import pandas as pd

apply_mobile_styles()

st.title("📊 رادار نخبگان v14.5")

# --- بخش رادار (جدول موبایلی) ---
if st.button("🔄 بروزرسانی رادار بازار"):
    radar_results = []
    with st.spinner("در حال پایش بازار..."):
        for coin in ["BTC", "ETH", "XRP"]:
            for tf in ["15m", "1h", "4h", "1d"]:
                data = get_data(coin, tf)
                if data is not None:
                    data = add_indicators(data)
                    power = get_ml_prediction(data)
                    sig = "BUY 🟩" if power > 65 else "SELL 🟥" if power < 35 else "WAIT ⚪"
                    radar_results.append({"ارز": coin, "تایم": tf, "سیگنال": sig, "قدرت": f"{power}%"})
    
    # نمایش به صورت کارت‌های عمودی برای موبایل (به جای جدول افقی)
    for res in radar_results:
        st.markdown(f"""
            <div class="radar-box">
                <b>{res['ارز']} ({res['تایم']})</b> | {res['سیگنال']} | قدرت: {res['قدرت']}
            </div>
        """, unsafe_allow_html=True)

st.divider()

# --- بخش تحلیل تک ارز ---
coin = st.selectbox("انتخاب ارز برای تحلیل عمیق:", list(CRYPTOS.keys()))
if st.button("🚀 تحلیل هوشمند"):
    df = get_data(coin, "1h")
    if df is not None:
        df = add_indicators(df)
        power = get_ml_prediction(df)
        st.metric("قدرت پیش‌بینی هوش مصنوعی", f"{power}%")
        # اینجا بقیه منطق مدیریت سرمایه را اضافه کنید
