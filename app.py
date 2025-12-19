import streamlit as st
import streamlit.components.v1 as components
from connector import get_data, CRYPTOS
from engine import add_indicators, calculate_trade_details
from styles import apply_styles
from datetime import datetime
import gc

# اعمال استایل‌های پایه
apply_styles()

st.title("💎 AI-CRYPTO ELITE v13.0")

with st.sidebar:
    st.header("👤 تنظیمات پنل")
    capital = st.number_input("سرمایه ($)", value=1000)
    risk = st.slider("ریسک (%)", 1.0, 5.0, 2.0)

coin = st.selectbox("ارز دیجیتال:", list(CRYPTOS.keys()))
tf = st.selectbox("تایم‌فریم:", ["15m", "1h", "4h", "1d"])

if st.button("🚀 اسکن و صدور سیگنال هوشمند"):
    with st.spinner("در حال تحلیل..."):
        df_raw = get_data(coin, tf)
        if df_raw is not None:
            df = add_indicators(df_raw)
            res = calculate_trade_details(df, capital, risk)
            
            is_long = "LONG" in res['sig']
            color = "#00cec9" if is_long else "#ff7675"
            bg = "rgba(0, 206, 201, 0.1)" if is_long else "rgba(255, 118, 117, 0.1)"
            emoji = "🟩" if is_long else "🟥"

            # ساخت بدنه گرافیکی
            html_content = f"""
            <div style="font-family: 'Tahoma', sans-serif; direction: rtl; background: #161a1e; border: 1px solid #2b2f36; border-radius: 15px; padding: 20px; color: white;">
                <h2 style="text-align:center; color:#f0b90b; margin:0;">📊 گزارش تحلیل هوشمند</h2>
                <p style="text-align:center; color:#848e9c; font-size:12px;">{datetime.now().strftime('%H:%M:%S')} | {coin}/USDT</p>
                
                <div style="background: {bg}; padding: 15px; border-radius: 12px; border-right: 5px solid {color}; margin: 15px 0;">
                    <div style="color:{color}; font-size:14px;">سیگنال:</div>
                    <div style="font-size:24px; font-weight:bold; color:{color};">{res['sig']} {emoji}</div>
                </div>

                <div style="display:flex; justify-content:space-between; margin:10px 0;">
                    <span style="color:#848e9c;">💵 قیمت ورود:</span>
                    <span style="font-weight:bold;">${res['price']:,.2f}</span>
                </div>

                <div style="background:rgba(255,118,117,0.1); padding:10px; border-radius:8px; display:flex; justify-content:space-between; border:1px dashed #ff7675;">
                    <span style="color:#ff7675;">🛡️ حد ضرر (SL):</span>
                    <span style="color:#ff7675; font-weight:bold;">${res['sl']:,.2f}</span>
                </div>

                <div style="margin-top:10px;">
                    <div style="display:flex; justify-content:space-between; background:rgba(255,255,255,0.05); padding:8px; border-radius:5px; margin-bottom:5px;">
                        <span>🎯 هدف اول:</span><span>${res['tp1']:,.2f}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; background:rgba(255,255,255,0.05); padding:8px; border-radius:5px;">
                        <span>🎯 هدف دوم:</span><span>${res['tp2']:,.2f}</span>
                    </div>
                </div>

                <div style="margin-top:20px; background:rgba(240,185,11,0.1); padding:15px; border-radius:10px; border:1px solid #f0b90b;">
                    <div style="color:#f0b90b; font-weight:bold; margin-bottom:5px;">🏛️ مدیریت سرمایه:</div>
                    <div style="display:flex; justify-content:space-between;">
                        <span>📏 حجم معامله:</span><span style="color:#f0b90b;">${res['pos_size']:,.2f}</span>
                    </div>
                </div>
            </div>
            """
            
            # استفاده از کامپوننت برای رندر اجباری HTML
            components.html(html_content, height=450, scrolling=False)
            
            st.success(f"تحلیل {coin} با موفقیت انجام شد.")
        else:
            st.error("خطا در دریافت اطلاعات")

gc.collect()
