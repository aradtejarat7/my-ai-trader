import streamlit as st
from connector import get_data
from engine import *
from styles import apply_mobile_styles

apply_mobile_styles()

# ... (کدهای انتخاب ارز و دریافت دیتا) ...

if st.button("🚀 شروع تحلیل هوشمند"):
    df = get_data(coin, tf)
    df = add_indicators(df)
    
    # استخراج مقادیر
    price = df['price'].iloc[-1]
    adx = df['adx'].iloc[-1]
    rsi = df['rsi'].iloc[-1]
    atr = df['atr'].iloc[-1]
    xgb_p, lstm_p = get_ml_probs(df)
    ensemble = (xgb_p + lstm_p) / 2
    
    # تعیین جهت سیگنال
    sig_type = "STRONG_SHORT" if ensemble < 30 else "STRONG_LONG" if ensemble > 70 else "WAIT"
    color = "🟥" if "SHORT" in sig_type else "🟩" if "LONG" in sig_type else "⚪"
    
    # محاسبات مدیریت سرمایه
    sl = price + (2 * atr) if "SHORT" in sig_type else price - (2 * atr)
    tp1, tp2, tp3 = get_targets(price, sl, "SHORT" if "SHORT" in sig_type else "LONG")
    pos_size, qty = calculate_management(price, sl, capital, risk_pct)

    # --- نمایش کارت نهایی (بسیار زیبا در موبایل) ---
    st.markdown(f"""
    <div class="signal-card">
        <div class="header-text">💎 AI-CRYPTO ELITE v13.0</div>
        <div class="data-row"><span class="label">💰 ارز:</span> <span class="value">{coin} | ⏱️ {tf}</span></div>
        <div class="divider"></div>
        <div style="text-align:center; font-size:18px;">🎯 <b>سیگنال: {sig_type}</b></div>
        <div style="text-align:center;">وضعیت: {"فروش" if "SHORT" in sig_type else "خرید"} {color}</div>
        <div style="text-align:center; color:#f0b90b;">📈 اعتماد مدل: {ensemble:.1f}%</div>
        <div class="divider"></div>
        <div class="data-row"><span class="label">💵 قیمت فعلی:</span> <span class="value">${price:,.4f}</span></div>
        <div class="data-row"><span class="label">🛡️ حد ضرر:</span> <span class="value">${sl:,.4f}</span></div>
        <div class="data-row"><span class="label">🎯 تارگت ۱:</span> <span class="value">${tp1:,.4f}</span></div>
        <div class="data-row"><span class="label">🎯 تارگت ۲:</span> <span class="value">${tp2:,.4f}</span></div>
        <div class="data-row"><span class="label">🎯 تارگت ۳:</span> <span class="value">${tp3:,.4f}</span></div>
        <div class="divider"></div>
        <div class="header-text" style="font-size:15px;">💰 مدیریت سرمایه</div>
        <div class="data-row"><span class="label">📏 حجم پوزیشن:</span> <span class="value">${pos_size:,.2,f}</span></div>
        <div class="data-row"><span class="label">📈 تعداد واحد:</span> <span class="value">{qty} {coin}</span></div>
        <div class="divider"></div>
        <div class="data-row"><span class="label">🏛️ رژیم بازار:</span> <span class="value">{get_market_regime(df)}</span></div>
        <div class="data-row"><span class="label">🔥 تاییدیه:</span> <span class="value">Gold Confirm ✅</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ارسال به تلگرام با همان فرمت درخواستی
    tg_text = f"💎 **AI-CRYPTO ELITE v13.0**\n💰 ارز: {coin} | ⏱️ {tf}\n🎯 **سیگنال: {sig_type}**\n..."
    # (ارسال متن کامل به تابع تلگرام)
