import streamlit as st
from connector import get_data, CRYPTOS
from engine import add_indicators, calculate_trade_details
from styles import apply_styles
from datetime import datetime
import gc

# اعمال استایل‌های گرافیکی
apply_styles()

# هدر اصلی اپلیکیشن
st.markdown('<h1 style="text-align:center; color:#f0b90b;">💎 AI-CRYPTO ELITE v13.0</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.header("👤 تنظیمات پنل")
    capital = st.number_input("سرمایه ($)", value=1000)
    risk = st.slider("ریسک (%)", 1.0, 5.0, 2.0)
    st.divider()
    st.caption("Developed by AI Elite Engine")

# انتخاب ارز و تایم‌فریم در یک ردیف
col_selection1, col_selection2 = st.columns(2)
with col_selection1:
    coin = st.selectbox("🎯 انتخاب ارز:", list(CRYPTOS.keys()))
with col_selection2:
    tf = st.selectbox("⏱️ تایم‌فریم:", ["15m", "1h", "4h", "1d"])

if st.button("🚀 شروع پردازش و اسکن بازار"):
    with st.spinner("در حال تحلیل لایه‌های شبکه عصبی..."):
        df_raw = get_data(coin, tf)
        if df_raw is not None:
            df = add_indicators(df_raw)
            res = calculate_trade_details(df, capital, risk)
            
            # منطق رنگ‌بندی پویا
            is_long = "LONG" in res['sig']
            color_theme = "#00cec9" if is_long else "#ff7675"
            bg_gradient = "rgba(0, 206, 201, 0.15)" if is_long else "rgba(255, 118, 117, 0.15)"
            emoji = "🟩" if is_long else "🟥"

            # نمایش کارت گرافیکی (اصلاح شده برای جلوگیری از نمایش کد خام)
            st.markdown(f"""
            <div class="signal-card">
                <div class="elite-header">📊 گزارش نهایی تحلیل هوشمند</div>
                <div style="text-align: center; color: #848e9c; font-size: 12px; margin-bottom: 15px;">
                    بروزرسانی: {datetime.now().strftime('%H:%M:%S')}
                </div>

                <div class="info-row">
                    <span class="info-label">💰 دارایی</span>
                    <span class="info-value">{coin} / USDT</span>
                </div>

                <div style="background: {bg_gradient}; padding: 18px; border-radius: 15px; margin: 15px 0; border-left: 5px solid {color_theme}; border-right: 5px solid {color_theme};">
                    <div style="font-size: 13px; color: {color_theme}; font-weight: bold;">🎯 پیشنهاد سیستم:</div>
                    <div style="font-size: 24px; font-weight: 900; color: {color_theme}; letter-spacing: 1px;">
                        {res['sig']} {emoji}
                    </div>
                    <div style="font-size: 13px; color: #ffffff; margin-top: 5px;">
                        سطح اطمینان مدل: <b>{res['adx']:.1f}%</b>
                    </div>
                </div>

                <div class="info-row">
                    <span class="info-label">💵 قیمت ورود (Entry)</span>
                    <span class="info-value" style="font-size: 20px; color: #f0b90b;">${res['price']:,.4f}</span>
                </div>

                <div class="stop-box" style="background: rgba(255, 118, 117, 0.1); border: 1px dashed #ff7675; padding: 12px; border-radius: 10px; display: flex; justify-content: space-between; margin-top: 10px;">
                    <span style="color: #ff7675; font-weight: bold;">🛡️ حد ضرر (SL)</span>
                    <span style="color: #ff7675; font-weight: bold;">${res['sl']:,.4f}</span>
                </div>

                <div style="margin-top: 15px;">
                    <div class="target-box" style="display: flex; justify-content: space-between; background: rgba(0, 206, 201, 0.05); padding: 8px 12px; border-radius: 8px; margin-bottom: 5px; border: 1px solid rgba(0, 206, 201, 0.2);">
                        <span style="color: #00cec9;">🎯 هدف اول (TP 1)</span>
                        <span style="font-weight: bold;">${res['tp1']:,.4f}</span>
                    </div>
                    <div class="target-box" style="display: flex; justify-content: space-between; background: rgba(0, 206, 201, 0.08); padding: 8px 12px; border-radius: 8px; margin-bottom: 5px; border: 1px solid rgba(0, 206, 201, 0.2);">
                        <span style="color: #00cec9;">🎯 هدف دوم (TP 2)</span>
                        <span style="font-weight: bold;">${res['tp2']:,.4f}</span>
                    </div>
                    <div class="target-box" style="display: flex; justify-content: space-between; background: rgba(0, 206, 201, 0.12); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(0, 206, 201, 0.2);">
                        <span style="color: #00cec9;">🎯 هدف سوم (TP 3)</span>
                        <span style="font-weight: bold;">${res['tp3']:,.4f}</span>
                    </div>
                </div>

                <div style="margin-top: 25px; padding: 18px; background: rgba(240, 185, 11, 0.05); border-radius: 15px; border: 1px solid rgba(240, 185, 11, 0.2);">
                    <div style="color: #f0b90b; font-size: 15px; margin-bottom: 12px; font-weight: bold; border-bottom: 1px solid rgba(240, 185, 11, 0.2); padding-bottom: 5px;">🏛️ مدیریت سرمایه نخبگان:</div>
                    <div class="info-row">
                        <span class="info-label">📏 حجم کل پوزیشن</span>
                        <span class="info-value" style="color: #f0b90b; font-size: 18px;">${res['pos_size']:,.2f} USDT</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">📈 مقدار واحد خرید</span>
                        <span class="info-value">{res['qty']:.4f} {coin}</span>
                    </div>
                </div>

                <div style="margin-top: 20px; display: flex; justify-content: space-around; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 15px;">
                    <div style="text-align: center;">
                        <div style="color: #848e9c; font-size: 11px;">ADX</div>
                        <div style="color: #ffffff; font-weight: bold; font-size: 16px;">{res['adx']:.1f}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #848e9c; font-size: 11px;">RSI</div>
                        <div style="color: #ffffff; font-weight: bold; font-size: 16px;">{res['rsi']:.1f}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #848e9c; font-size: 11px;">وضعیت</div>
                        <div style="color: #f0b90b; font-weight: bold; font-size: 13px;">GOLD ✅</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.toast(f"تحلیل {coin} با موفقیت به پایان رسید", icon='✅')
            
        else:
            st.error("❌ عدم پاسخگویی صرافی. لطفاً اینترنت خود یا تایم‌فریم را چک کنید.")

# مدیریت حافظه
gc.collect()
