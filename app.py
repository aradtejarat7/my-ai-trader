import streamlit as st
from connector import get_data, CRYPTOS
from engine import add_indicators, calculate_trade_details
from styles import apply_styles
from datetime import datetime

# اعمال استایل‌های گرافیکی جدید
apply_styles()

st.title("💎 AI-CRYPTO ELITE v13.0")

with st.sidebar:
    st.header("👤 تنظیمات پنل")
    capital = st.number_input("سرمایه ($)", value=1000)
    risk = st.slider("ریسک (%)", 1.0, 5.0, 2.0)
    st.divider()
    st.caption("Developed by AI Elite Engine")

# انتخاب ارز و تایم‌فریم
col_selection1, col_selection2 = st.columns(2)
with col_selection1:
    coin = st.selectbox("ارز دیجیتال:", list(CRYPTOS.keys()))
with col_selection2:
    tf = st.selectbox("تایم‌فریم:", ["15m", "1h", "4h", "1d"])

if st.button("🚀 اسکن و صدور سیگنال هوشمند"):
    with st.spinner("در حال تحلیل لایه‌های شبکه عصبی و دریافت دیتا..."):
        df_raw = get_data(coin, tf)
        if df_raw is not None:
            df = add_indicators(df_raw)
            res = calculate_trade_details(df, capital, risk)
            
            # تعیین رنگ و جهت بر اساس سیگنال موجود در res
            is_long = "LONG" in res['sig']
            color_theme = "#00cec9" if is_long else "#ff7675"
            bg_color = "rgba(0, 206, 201, 0.1)" if is_long else "rgba(255, 118, 117, 0.1)"
            emoji = "🟩" if is_long else "🟥"

            # نمایش کارت گرافیکی بهبود یافته
            st.markdown(f"""
            <div class="signal-card">
                <div class="elite-header">💎 AI-CRYPTO ELITE v13.0</div>
                <div style="text-align: center; color: #848e9c; font-size: 12px; margin-bottom: 15px;">
                    ⏰ {datetime.now().strftime('%H:%M:%S')} | ⏱️ {tf}
                </div>

                <div class="info-row">
                    <span class="info-label">💰 جفت ارز</span>
                    <span class="info-value">{coin} / USDT</span>
                </div>

                <div style="background: {bg_color}; padding: 15px; border-radius: 12px; margin: 15px 0; border-right: 5px solid {color_theme};">
                    <div style="font-size: 13px; color: {color_theme};">🎯 سیگنال صادر شده:</div>
                    <div style="font-size: 22px; font-weight: 800; color: {color_theme};">
                        {res['sig']} {emoji}
                    </div>
                    <div style="font-size: 12px; color: #ffffff;">اعتماد مدل: {res['adx']:.1f}%</div>
                </div>

                <div class="info-row">
                    <span class="info-label">💵 قیمت فعلی (Entry)</span>
                    <span class="info-value" style="font-size: 18px;">${res['price']:,.4f}</span>
                </div>

                <div class="stop-box">
                    <span style="color: #ff7675; font-weight: bold;">🛡️ حد ضرر (Stop Loss)</span>
                    <span style="color: #ff7675; font-weight: bold;">${res['sl']:,.4f}</span>
                </div>

                <div style="margin-top: 10px;">
                    <div class="target-box">
                        <span>🎯 هدف اول (TP 1)</span>
                        <span>${res['tp1']:,.4f}</span>
                    </div>
                    <div class="target-box">
                        <span>🎯 هدف دوم (TP 2)</span>
                        <span>${res['tp2']:,.4f}</span>
                    </div>
                    <div class="target-box">
                        <span>🎯 هدف سوم (TP 3)</span>
                        <span>${res['tp3']:,.4f}</span>
                    </div>
                </div>

                <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.03); border-radius: 12px;">
                    <div style="color: #f0b90b; font-size: 14px; margin-bottom: 10px; font-weight: bold;">🏛️ مدیریت سرمایه:</div>
                    <div class="info-row">
                        <span class="info-label">📏 حجم پوزیشن</span>
                        <span class="info-value" style="color: #f0b90b;">${res['pos_size']:,.2f}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">📈 تعداد واحد</span>
                        <span class="info-value">{res['qty']:.4f} {coin}</span>
                    </div>
                </div>

                <div style="margin-top: 15px; display: flex; justify-content: space-around; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 10px;">
                    <div style="text-align: center;">
                        <div style="color: #848e9c; font-size: 10px;">ADX</div>
                        <div style="color: #ffffff; font-weight: bold;">{res['adx']:.1f}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #848e9c; font-size: 10px;">RSI</div>
                        <div style="color: #ffffff; font-weight: bold;">{res['rsi']:.1f}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #848e9c; font-size: 10px;">CONFIRM</div>
                        <div style="color: #f0b90b; font-weight: bold;">GOLD ✅</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # افکت موفقیت برای تجربه کاربری بهتر
            st.toast(f"سیگنال {coin} با موفقیت صادر شد", icon='🚀')
            
        else:
            st.error("❌ خطا در دریافت دیتا از صرافی. لطفا دوباره تلاش کنید.")

# پاکسازی حافظه در پایان اجرا
import gc
gc.collect()
