import streamlit as st
from connector import get_data, CRYPTOS
from engine import add_indicators, calculate_trade_details
from styles import apply_styles
from datetime import datetime
import gc

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

            # کل محتوای کارت را در یک متغیر ذخیره می‌کنیم تا تداخلی ایجاد نشود
            card_html = f"""
            <div class="signal-card" style="background: #161a1e; border: 1px solid #2b2f36; border-radius: 15px; padding: 20px; direction: rtl; text-align: right;">
                <div class="elite-header" style="text-align:center; color:#f0b90b; font-weight:bold; font-size:18px; margin-bottom:10px;">📊 گزارش نهایی تحلیل هوشمند</div>
                <div style="text-align: center; color: #848e9c; font-size: 12px; margin-bottom: 15px;">
                    ⏰ {datetime.now().strftime('%H:%M:%S')} | ⏱️ {tf}
                </div>

                <div class="info-row" style="display: flex; justify-content: space-between; margin-bottom: 10px; border-bottom: 1px solid #2b2f36; padding-bottom: 5px;">
                    <span class="info-label" style="color: #848e9c;">💰 جفت ارز</span>
                    <span class="info-value" style="color: white; font-weight: bold;">{coin} / USDT</span>
                </div>

                <div style="background: {bg_color}; padding: 15px; border-radius: 12px; margin: 15px 0; border-right: 5px solid {color_theme};">
                    <div style="font-size: 13px; color: {color_theme};">🎯 سیگنال صادر شده:</div>
                    <div style="font-size: 22px; font-weight: 800; color: {color_theme};">
                        {res['sig']} {emoji}
                    </div>
                    <div style="font-size: 12px; color: #ffffff;">اعتماد مدل: {res['adx']:.1f}%</div>
                </div>

                <div class="info-row" style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span class="info-label" style="color: #848e9c;">💵 قیمت ورود (Entry)</span>
                    <span class="info-value" style="font-size: 18px; color: white; font-weight: bold;">${res['price']:,.4f}</span>
                </div>

                <div class="stop-box" style="background: rgba(255, 118, 117, 0.1); border: 1px dashed #ff7675; padding: 12px; border-radius: 10px; display: flex; justify-content: space-between; margin-top: 10px;">
                    <span style="color: #ff7675; font-weight: bold;">🛡️ حد ضرر (SL)</span>
                    <span style="color: #ff7675; font-weight: bold;">${res['sl']:,.4f}</span>
                </div>

                <div style="margin-top: 15px;">
                    <div class="target-box" style="display: flex; justify-content: space-between; background: rgba(0, 206, 201, 0.05); padding: 8px 12px; border-radius: 8px; margin-bottom: 5px; border: 1px solid rgba(0, 206, 201, 0.2);">
                        <span style="color: #00cec9;">🎯 هدف اول (TP 1)</span>
                        <span style="color: white; font-weight: bold;">${res['tp1']:,.4f}</span>
                    </div>
                    <div class="target-box" style="display: flex; justify-content: space-between; background: rgba(0, 206, 201, 0.08); padding: 8px 12px; border-radius: 8px; margin-bottom: 5px; border: 1px solid rgba(0, 206, 201, 0.2);">
                        <span style="color: #00cec9;">🎯 هدف دوم (TP 2)</span>
                        <span style="color: white; font-weight: bold;">${res['tp2']:,.4f}</span>
                    </div>
                    <div class="target-box" style="display: flex; justify-content: space-between; background: rgba(0, 206, 201, 0.12); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(0, 206, 201, 0.2);">
                        <span style="color: #00cec9;">🎯 هدف سوم (TP 3)</span>
                        <span style="color: white; font-weight: bold;">${res['tp3']:,.4f}</span>
                    </div>
                </div>

                <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.03); border-radius: 12px; border: 1px solid #2b2f36;">
                    <div style="color: #f0b90b; font-size: 14px; margin-bottom: 10px; font-weight: bold;">🏛️ مدیریت سرمایه نخبگان:</div>
                    <div class="info-row" style="display: flex; justify-content: space-between;">
                        <span class="info-label" style="color: #848e9c;">📏 حجم پوزیشن</span>
                        <span class="info-value" style="color: #f0b90b; font-weight: bold;">${res['pos_size']:,.2f} USDT</span>
                    </div>
                    <div class="info-row" style="display: flex; justify-content: space-between;">
                        <span class="info-label" style="color: #848e9c;">📈 مقدار واحد</span>
                        <span class="info-value" style="color: white; font-weight: bold;">{res['qty']:.4f} {coin}</span>
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
            """
            
            # نمایش نهایی متغیر HTML با اجازه اجرای کدها
            st.markdown(card_html, unsafe_allow_html=True)
            
            # افکت موفقیت
            st.toast(f"سیگنال {coin} با موفقیت صادر شد", icon='🚀')
            
        else:
            st.error("❌ خطا در دریافت دیتا از صرافی. لطفا دوباره تلاش کنید.")

# پاکسازی حافظه
gc.collect()
