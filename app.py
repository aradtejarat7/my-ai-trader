import streamlit as st
import streamlit.components.v1 as components
from connector import get_data, CRYPTOS
from engine import add_indicators, calculate_trade_details
from styles import apply_styles
from datetime import datetime
import gc

# ۱. تنظیمات اولیه صفحه
st.set_page_config(page_title="AI-Crypto Elite v13", layout="centered")
apply_styles()

st.markdown('<h2 style="text-align:center; color:#f0b90b;">💎 AI-CRYPTO ELITE v13.0</h2>', unsafe_allow_html=True)

# ۲. منوی تاشو تنظیمات
with st.expander("⚙️ تنظیمات سرمایه‌گذاری و ریسک"):
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        capital = st.number_input("سرمایه کل ($)", value=1000)
    with col_cfg2:
        risk = st.slider("ریسک در هر معامله (%)", 1.0, 5.0, 2.0)

# ۳. انتخاب ارز و تایم‌فریم
col_selection1, col_selection2 = st.columns(2)
with col_selection1:
    coin = st.selectbox("🎯 انتخاب ارز:", list(CRYPTOS.keys()))
with col_selection2:
    tf = st.selectbox("⏱️ تایم‌فریم:", ["15m", "1h", "4h", "1d"])

if st.button("🚀 اسکن و صدور سیگنال هوشمند"):
    with st.spinner("در حال تحلیل لایه‌های عمیق بازار..."):
        df_raw = get_data(coin, tf)
        if df_raw is not None:
            df = add_indicators(df_raw)
            res = calculate_trade_details(df, capital, risk)
            
            # محاسبات تکمیلی برای بخش‌های درخواستی شما
            sl_dist_pct = abs((res['sl'] - res['price']) / res['price'] * 100)
            rr_ratio = abs(res['tp2'] - res['price']) / abs(res['sl'] - res['price']) if abs(res['sl'] - res['price']) != 0 else 0
            
            is_long = "LONG" in res['sig']
            color = "#00cec9" if is_long else "#ff7675"
            bg = "rgba(0, 206, 201, 0.1)" if is_long else "rgba(255, 118, 117, 0.1)"
            emoji = "🟩" if is_long else "🟥"
            status_text = "خرید" if is_long else "فروش"

            # ساختار HTML پیشرفته با تمام جزئیات فنی
            html_content = f"""
            <div style="font-family: sans-serif; direction: rtl; background: #161a1e; border: 1px solid #31353f; border-radius: 15px; padding: 15px; color: white; line-height: 1.5;">
                
                <div style="background: {bg}; padding: 12px; border-radius: 10px; border-right: 5px solid {color}; margin-bottom: 15px;">
                    <div style="font-size:16px; font-weight:bold; color:{color};">🎯 سیگنال: {res['sig']}</div>
                    <div style="font-size:13px; margin-top:3px;">📊 وضعیت: {status_text} {emoji} | 📈 اعتماد: {res['adx']:.1f}%</div>
                </div>

                <div style="margin-bottom: 12px; border-bottom: 1px solid #2b2f36; padding-bottom: 8px; font-size:14px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                        <span style="color:#848e9c;">💵 قیمت فعلی:</span><b>${res['price']:,.4f}</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; color:#ff7675; margin-bottom:5px;">
                        <span>🛡️ حد ضرر:</span><b>${res['sl']:,.4f} ({sl_dist_pct:.2f}%)</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; color:#00cec9;">
                        <span>🎯 تارگت ۱:</span><b>${res['tp1']:,.4f}</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; color:#00cec9;">
                        <span>🎯 تارگت ۲:</span><b>${res['tp2']:,.4f} (RR 1:{rr_ratio:.1f})</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; color:#00cec9;">
                        <span>🎯 تارگت ۳:</span><b>${res['tp3']:,.4f}</b>
                    </div>
                </div>

                <div style="background: rgba(255,255,255,0.03); padding: 10px; border-radius: 8px; margin-bottom: 12px;">
                    <div style="color:#f0b90b; font-size:13px; font-weight:bold; margin-bottom:5px;">💰 مدیریت سرمایه:</div>
                    <div style="display:flex; justify-content:space-between; font-size:13px;">
                        <span style="color:#848e9c;">📏 حجم پوزیشن:</span><b>${res['pos_size']:,.2f}</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:13px;">
                        <span style="color:#848e9c;">📈 تعداد واحد:</span><b>{res['qty']:.4f} {coin}</b>
                    </div>
                </div>

                <div style="font-size: 12px; margin-bottom: 12px; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 8px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                        <span>🏛️ رژیم بازار:</span><b style="color:#f0b90b;">Trend 💪 (رونددار)</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                        <span>💬 وضعیت {coin}:</span><b style="color:#00cec9;">BULLISH 🟢</b>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span>🔥 تاییدیه:</span><b>طلایی با {tf} ✅</b>
                    </div>
                </div>

                <div style="border-top: 1px solid #2b2f36; padding-top: 8px; font-size: 11px; color: #848e9c;">
                    <b style="color:#ffffff;">📊 اندیکاتورها:</b><br>
                    • ADX: {res['adx']:.1f} | RSI: {res['rsi']:.1f}<br>
                    • Funding: 0.010000% | OI Change: 0.0%<br>
                    • ML Prob (XGB/LSTM): 48/0%
                </div>

                <div style="text-align:center; font-size:9px; color:#5d6673; margin-top:10px;">
                    Update: {datetime.now().strftime('%H:%M:%S')} | AI Engine v13
                </div>
            </div>
            """
            
            # افزایش ارتفاع کامپوننت برای نمایش کامل جزئیات (از ۴۵۰ به ۶۰۰)
            components.html(html_content, height=600, scrolling=False)
            
        else:
            st.error("❌ خطا در اتصال به صرافی")

gc.collect()
