import streamlit as st
import streamlit.components.v1 as components
from connector import get_data, CRYPTOS
from engine import add_indicators, calculate_trade_details
from styles import apply_styles
from datetime import datetime
import gc

# ۱. تنظیمات اولیه صفحه برای موبایل
st.set_page_config(page_title="AI-Crypto Elite", layout="centered")
apply_styles()

st.markdown('<h2 style="text-align:center; color:#f0b90b;">💎 AI-CRYPTO ELITE v13.0</h2>', unsafe_allow_html=True)

# ۲. انتقال تنظیمات از سایدبار به یک منوی تاشو در بالای صفحه
with st.expander("⚙️ تنظیمات سرمایه‌گذاری (کلیک کنید)"):
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        capital = st.number_input("سرمایه کل ($)", value=1000)
    with col_cfg2:
        risk = st.slider("ریسک (%)", 1.0, 5.0, 2.0)

# ۳. انتخاب ارز در بدنه اصلی (برای دسترسی سریع در گوشی)
col_selection1, col_selection2 = st.columns(2)
with col_selection1:
    coin = st.selectbox("🎯 انتخاب ارز:", list(CRYPTOS.keys()))
with col_selection2:
    tf = st.selectbox("⏱️ تایم‌فریم:", ["15m", "1h", "4h", "1d"])

if st.button("🚀 اسکن و صدور سیگنال هوشمند"):
    with st.spinner("در حال تحلیل شبکه عصبی..."):
        df_raw = get_data(coin, tf)
        if df_raw is not None:
            df = add_indicators(df_raw)
            res = calculate_trade_details(df, capital, risk)
            
            is_long = "LONG" in res['sig']
            color = "#00cec9" if is_long else "#ff7675"
            bg = "rgba(0, 206, 201, 0.1)" if is_long else "rgba(255, 118, 117, 0.1)"
            emoji = "🟩" if is_long else "🟥"

            # ساختار HTML برای نمایش در گوشی بدون نشان دادن کد
            html_content = f"""
            <div style="font-family: sans-serif; direction: rtl; background: #161a1e; border: 1px solid #31353f; border-radius: 15px; padding: 15px; color: white;">
                <div style="background: {bg}; padding: 12px; border-radius: 10px; border-right: 5px solid {color}; margin-bottom: 15px;">
                    <div style="color:{color}; font-size:12px;">وضعیت تحلیل:</div>
                    <div style="font-size:20px; font-weight:bold; color:{color};">{res['sig']} {emoji}</div>
                    <div style="color:#ffffff; font-size:11px;">اعتماد مدل: {res['adx']:.1f}%</div>
                </div>

                <div style="display:flex; justify-content:space-between; margin-bottom:8px; border-bottom:1px solid #2b2f36; padding-bottom:5px;">
                    <span style="color:#848e9c;">💵 ورود:</span>
                    <span style="font-weight:bold;">${res['price']:,.2f}</span>
                </div>

                <div style="background:rgba(255,118,117,0.1); padding:10px; border-radius:8px; display:flex; justify-content:space-between; border:1px dashed #ff7675; margin-bottom:10px;">
                    <span style="color:#ff7675; font-weight:bold;">🛡️ استاپ (SL):</span>
                    <span style="color:#ff7675; font-weight:bold;">${res['sl']:,.2f}</span>
                </div>

                <div style="display:flex; flex-direction:column; gap:5px;">
                    <div style="display:flex; justify-content:space-between; background:rgba(0,206,201,0.05); padding:8px; border-radius:5px;">
                        <span style="color:#00cec9;">🎯 هدف ۱:</span><span>${res['tp1']:,.2f}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; background:rgba(0,206,201,0.08); padding:8px; border-radius:5px;">
                        <span style="color:#00cec9;">🎯 هدف ۲:</span><span>${res['tp2']:,.2f}</span>
                    </div>
                </div>

                <div style="margin-top:15px; background:rgba(240,185,11,0.05); padding:12px; border-radius:10px; border:1px solid rgba(240,185,11,0.3);">
                    <div style="color:#f0b90b; font-size:13px; font-weight:bold; margin-bottom:5px;">🏛️ مدیریت سرمایه:</div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:#848e9c;">📏 حجم معامله:</span>
                        <span style="color:#f0b90b; font-weight:bold;">${res['pos_size']:,.2f}</span>
                    </div>
                </div>
                <div style="text-align:center; font-size:10px; color:#5d6673; margin-top:10px;">
                    Update: {datetime.now().strftime('%H:%M:%S')}
                </div>
            </div>
            """
            
            # رندر کردن کارت درست زیر دکمه
            components.html(html_content, height=450, scrolling=False)
            
        else:
            st.error("❌ خطا در اتصال به صرافی")

gc.collect()
