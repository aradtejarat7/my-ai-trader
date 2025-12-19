# سلول اول: نصب کتابخانه‌ها و پیکربندی تنظیمات
# نصب کتابخانه‌های مورد نیاز (ta برای اندیکاتورها و nest_asyncio برای اجرای async در کولب)
!pip install ta nest_asyncio -q

import asyncio
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime

# کتابخانه‌های یادگیری ماشین (Machine Learning)
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import backend as K # برای پاکسازی حافظه رم

# کتابخانه تحلیل تکنیکال
import ta

# فعال‌سازی محیط ناهمگام برای جلوگیری از فریز شدن در گوگل کولب
import nest_asyncio
nest_asyncio.apply()

# =========================
# تنظیمات اصلی (CONFIG)
# =========================
# توکن ربات تلگرام و آی‌دی چت شما
TOKEN = "8548739067:AAGuvMHgB-LxOoyQIrHWzs6ytTfOehfIrco"
CHAT_ID = "163583693"

# لیست ارزهایی که می‌خواهید تحلیل شوند (نام برای API و نماد برای نمایش)
CRYPTOS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "ripple": "XRP"
}

# تنظیمات زمانی و معاملاتی
INTERVAL_SECONDS = 300   # زمان بین هر تحلیل (۵ دقیقه)
CANDLES = 1000           # تعداد کندل‌های مورد بررسی برای آموزش مدل
POSITION_SIZE = 1500     # حجم فرضی هر معامله به دلار
RISK_REWARD = "1:1.20"   # نسبت ریسک به ریوارد برای نمایش در پیام


# سلول دوم: توابع ارتباطی، دریافت داده‌های زنده (Spot & Futures) و محاسبه اندیکاتورها

def send_telegram(text):
    """ارسال پیام متنی به تلگرام"""
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=30)
    except Exception as e:
        print(f"خطا در تلگرام: {e}")

def get_futures_info(symbol):
    """دریافت داده‌های Open Interest و Funding Rate از بازار فیوچرز بایننس"""
    try:
        # دریافت نرخ تامین مالی (Funding Rate)
        fund_url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
        fund_data = requests.get(fund_url, timeout=10).json()
        funding_rate = float(fund_data.get("lastFundingRate", 0)) * 100 # تبدیل به درصد

        # دریافت Open Interest
        oi_url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
        oi_data = requests.get(oi_url, timeout=10).json()
        open_interest = float(oi_data.get("openInterest", 0))

        return funding_rate, open_interest
    except:
        return 0.0, 0.0

def get_data(coin, interval="1h"): # تغییر پیش‌فرض به 1 ساعته طبق درخواست شما
    """دریافت داده‌های قیمت و حجم از بایننس"""
    symbol_map = {"bitcoin": "BTCUSDT", "ethereum": "ETHUSDT", "ripple": "XRPUSDT"}
    symbol = symbol_map.get(coin, "BTCUSDT")

    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={CANDLES}"

    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200: return None
        data = r.json()

        df = pd.DataFrame(data, columns=[
            "ts", "open", "high", "low", "close", "volume",
            "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"
        ])

        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df["price"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float) # اضافه شدن ستون حجم

        df.set_index("ts", inplace=True)

        # دریافت اطلاعات تکمیلی فیوچرز
        funding, oi = get_futures_info(symbol)
        df["funding_rate"] = funding
        df["open_interest"] = oi

        return df[["price", "high", "low", "volume", "funding_rate", "open_interest"]]
    except Exception as e:
        print(f"خطا در دریافت داده: {e}")
        return None

def add_indicators(df):
    """اضافه کردن شاخص‌های ارتقا یافته (RSI, MACD, ADX, ATR, Vol Ratio)"""
    try:
        # ۱. شاخص RSI
        df["rsi"] = ta.momentum.RSIIndicator(df["price"]).rsi()

        # ۲. شاخص MACD
        macd = ta.trend.MACD(df["price"])
        df["macd"] = macd.macd_diff()

        # ۳. شاخص ADX (تشخیص قدرت روند)
        adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["price"], window=14)
        df["adx"] = adx_ind.adx()

        # ۴. میانگین متحرک (EMA 20)
        df["ema"] = ta.trend.EMAIndicator(df["price"], 20).ema_indicator()

        # ۵. شاخص ATR (نوسان‌سنجی)
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["price"]).average_true_range()

        # ۶. نسبت حجم (Volume Ratio) - نسبت حجم فعلی به میانگین ۲۰ کندل اخیر
        df["vol_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # ۷. محاسبه تغییر Open Interest (نسبت به کندل قبل)
        df["oi_change"] = df["open_interest"].pct_change() * 100

        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"خطا در محاسبات تکنیکال: {e}")
        return df




# سلول سوم: موتور جامع تحلیل v13.0 - ترکیب کامل ML + مدیریت سرمایه + تاییدیه طلایی
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.backend as K
from datetime import datetime

# --- تنظیمات استراتژیک کاربر ---
MY_CAPITAL = 1000  # کل سرمایه شما به دلار
RISK_PER_TRADE = 0.02  # ریسک ۲ درصد در هر معامله
# ----------------------------

if 'signal_history' not in globals(): signal_history = {}
if 'signal_scores' not in globals(): signal_scores = {}
if 'market_regimes' not in globals(): market_regimes = {}

# ۱. تابع XGBoost برای پیش‌بینی احتمال حرکت قیمت
def train_xgb(df):
    try:
        features = ["rsi", "macd", "ema", "atr", "adx", "vol_ratio"]
        X = df[features].copy()
        y = (df["price"].shift(-1) > df["price"]).astype(int)
        X, y = X[:-1], y[:-1]
        model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, verbosity=0)
        model.fit(X, y)
        prob = model.predict_proba(X.iloc[-1:])[0][1]
        return prob * 100
    except: return 50

# ۲. تابع LSTM برای تحلیل سری‌های زمانی
def train_lstm(df):
    try:
        K.clear_session()
        data = df[["price"]].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        if len(scaled_data) < 51: return 50
        X_train = np.array([scaled_data[-51:-1]])
        model = Sequential([
            LSTM(16, input_shape=(50, 1)),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, np.array([scaled_data[-1]]), epochs=1, verbose=0)
        pred = model.predict(X_train, verbose=0)[0][0]
        return 100 if pred > scaled_data[-1][0] else 0
    except: return 50

# ۳. تحلیل جهت‌گیری کل بازار (BTC Bias)
def get_btc_bias():
    try:
        df_btc = get_data("BTC", interval="1h")
        if df_btc is None or df_btc.empty: return "UNKNOWN", "⚪"
        last_price = df_btc['price'].iloc[-1]
        ema_btc = df_btc['price'].ewm(span=50).mean().iloc[-1]
        bias = "BULLISH" if last_price > ema_btc else "BEARISH"
        emoji = "🟢" if bias == "BULLISH" else "🔴"
        return bias, emoji
    except: return "UNKNOWN", "⚪"

# ۴. موتور اصلی تحلیل و مدیریت معامله
def analyze(df, symbol, tf, coin_key):
    try:
        # ۱. آماده‌سازی داده‌های فنی
        row = df.iloc[-1]
        price = row["price"]
        rsi, adx, atr, ema = row["rsi"], row["adx"], row["atr"], row["ema"]
        vol_ratio = row.get("vol_ratio", 1.0)
        funding = row.get('funding_rate', 0.0)
        oi_change = row.get('oi_change', 0.0)

        # ۲. رژیم بازار
        if adx > 25: regime = "Trend 💪 (رونددار)"
        elif adx < 20: regime = "Range 💤 (ساید)"
        else: regime = "Stable ⚖️ (متعادل)"

        if coin_key not in market_regimes: market_regimes[coin_key] = {}
        market_regimes[coin_key][tf] = regime

        # ۳. اجرای مدل‌های هوش مصنوعی (Ensemble)
        xgb_p = train_xgb(df)
        lstm_p = train_lstm(df)
        trend_score = 100 if price > ema else 0
        ensemble_score = (xgb_p * 0.45) + (lstm_p * 0.40) + (trend_score * 0.15)

        # ۴. تعیین سیگنال نهایی و فیلتر BTC
        btc_bias, btc_emoji = get_btc_bias()
        signal_type = "NEUTRAL"
        if ensemble_score > 70 and adx > 18: signal_type = "STRONG_LONG"
        elif ensemble_score < 30 and adx > 18: signal_type = "STRONG_SHORT"

        # فیلتر همبستگی با بیت‌کوین
        final_signal = signal_type
        if "BTC" not in symbol and signal_type != "NEUTRAL":
            if (signal_type == "STRONG_LONG" and btc_bias == "BEARISH") or \
               (signal_type == "STRONG_SHORT" and btc_bias == "BULLISH"):
                final_signal = "WAIT_CONFIRM ⚠️"

        # ۵. تاییدیه طلایی (Multi-Timeframe Confirm)
        if coin_key not in signal_history: signal_history[coin_key] = {}
        signal_history[coin_key][tf] = "LONG" if "LONG" in final_signal else "SHORT" if "SHORT" in final_signal else "HOLD"

        is_golden = False
        gold_msg = ""
        for htf in ["1h", "4h", "1d"]:
            if htf in signal_history[coin_key] and htf != tf:
                if signal_history[coin_key][htf] == signal_history[coin_key][tf] and signal_history[coin_key][tf] != "HOLD":
                    is_golden = True
                    gold_msg = f"🔥 تاییدیه طلایی با {htf}"

        # ۶. مدیریت ریسک حرفه‌ای
        sl_dist = 2.5 * atr
        sl = price - sl_dist if "LONG" in final_signal else price + sl_dist
        sl_pct = (abs(price - sl) / price) * 100

        # محاسبه حجم ورود (Position Sizing)
        risk_amt = MY_CAPITAL * RISK_PER_TRADE
        qty = risk_amt / abs(price - sl) if abs(price - sl) != 0 else 0
        pos_size_usd = qty * price

        # تارگت‌های ۳ مرحله‌ای
        tp1 = price + (1.2 * atr) if "LONG" in final_signal else price - (1.2 * atr)
        tp2 = price + (2.8 * atr) if "LONG" in final_signal else price - (2.8 * atr)
        tp3 = price + (5.0 * atr) if "LONG" in final_signal else price - (5.0 * atr)
        rr_ratio = abs(tp2 - price) / abs(price - sl) if abs(price - sl) != 0 else 0

        # ۷. ذخیره امتیاز برای داشبورد
        global signal_scores
        if coin_key not in signal_scores: signal_scores[coin_key] = {}
        signal_scores[coin_key][tf] = ensemble_score

        # ۸. قالب‌بندی خروجی نهایی
        emoji_h = "💎" if is_golden else "🤖"
        status_txt = "صبر / خنثی ⬜️" if "NEUTRAL" in final_signal or "WAIT" in final_signal else ("خرید 🟩" if "LONG" in final_signal else "فروش 🟥")

        return f"""
{emoji_h} **AI-CRYPTO ELITE v13.0**
💰 ارز: {symbol.replace('USDT', '')} | ⏱️ {tf}
───────────────────
🎯 **سیگنال: {final_signal}**
📊 وضعیت: {status_txt}
📈 اعتماد مدل: {ensemble_score:.1f}%

💵 قیمت فعلی: ${price:.4f}
🛡️ حد ضرر: ${sl:.4f} ({sl_pct:.2f}%)
🎯 تارگت ۱: ${tp1:.4f}
🎯 تارگت ۲: ${tp2:.4f} (RR 1:{rr_ratio:.1f})
🎯 تارگت ۳: ${tp3:.4f}

💰 **مدیریت سرمایه:**
📏 حجم پوزیشن: ${pos_size_usd:.2f}
📈 تعداد واحد: {qty:.4f} {symbol.replace('USDT', '')}

🏛️ رژیم بازار: {regime}
💬 بیت‌کوین: {btc_bias} {btc_emoji}
{gold_msg if is_golden else "🔍 در انتظار تایید مولتی‌تایم‌فریم..."}

📊 **اندیکاتورها:**
• ADX: {adx:.1f} | RSI: {rsi:.1f}
• Funding: {funding:.6f}%
• OI Change: {oi_change:.1f}%
• ML Prob (XGB/LSTM): {xgb_p:.0f}/{lstm_p:.0f}%
───────────────────
"""
    except Exception as e:
        return f"❌ خطا در موتور تحلیل: {str(e)}"


# سلول نهایی v13.5: مدیریت زمان‌بندی + داشبورد همزمان کولب و تلگرام
from google.colab import output
import gc
import sys
import asyncio
import time
import pandas as pd
from datetime import datetime
from IPython.display import display, HTML

# متغیرهای جهانی برای ردیابی وضعیت
if 'signal_scores' not in globals(): signal_scores = {}
if 'signal_history' not in globals(): signal_history = {}
if 'market_regimes' not in globals(): market_regimes = {}

def show_dashboard():
    """نمایش داشبورد در کولب و ارسال نسخه متنی به تلگرام"""
    try:
        if not signal_history:
            display(HTML("<div style='color: #f1c40f; padding: 20px; font-family: Tahoma; background: #1a1a1a; border-radius: 12px; text-align: center; border: 1px dashed #f1c40f;'>⏳ سیستم در حال تحلیل مدل‌ها...</div>"))
            return

        btc_status, btc_icon = get_btc_bias()
        data_list = []

        # --- آماده‌سازی گزارش متنی برای تلگرام ---
        tg_msg = f"📊 **رادار نخبگان بازار v13.5**\n"
        tg_msg += f"وضعیت بیت‌کوین: {btc_icon} {btc_status}\n"
        tg_msg += "───────────────────\n"
        tg_msg += "`| ارز   | تایم | سیگنال  | قدرت |` \n"

        for coin_key, tfs in signal_history.items():
            for tf, sig in tfs.items():
                score = signal_scores.get(coin_key, {}).get(tf, 50.0)
                regime = market_regimes.get(coin_key, {}).get(tf, "Searching...")

                # تنظیم آیکون و لیبل برای هر دو پلتفرم
                if "LONG" in sig:
                    st_lbl, color, icon, bg = "LONG", "#2ecc71", "🟩", "rgba(46, 204, 113, 0.08)"
                elif "SHORT" in sig:
                    st_lbl, color, icon, bg = "SHORT", "#e74c3c", "🟥", "rgba(231, 76, 60, 0.08)"
                else:
                    st_lbl, color, icon, bg = "WAIT", "#f1c40f", "⏳", "transparent"

                coin_name = coin_key.upper().replace("USDT", "")

                # افزودن به لیست دیتای کولب
                data_list.append({
                    "ارز": coin_name, "تایم‌فریم": tf, "رژیم": regime,
                    "وضعیت": f"{st_lbl} {icon}", "قدرت": f"{score:.1f}%",
                    "color": color, "bg": bg, "score_val": score
                })

                # افزودن به متن تلگرام با فرمت ستونی ثابت (Monospace)
                tg_msg += f"`| {coin_name:<6} | {tf:<4} | {st_lbl:<7} | {score:>3.0f}% |` \n"

        tg_msg += "───────────────────\n"
        tg_msg += f"⏰ بروزرسانی: {datetime.now().strftime('%H:%M:%S')}"

        # ۱. ارسال به تلگرام
        send_telegram(tg_msg)

        # ۲. نمایش گرافیکی در گوگل کولب
        df_dash = pd.DataFrame(data_list).sort_values(by=["ارز", "تایم‌فریم"])
        html_table = f"""
        <div style="direction: rtl; font-family: 'Tahoma'; padding: 20px; background-color: #080808; border-radius: 15px; border: 1px solid #333;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h2 style="color: #f1c40f; margin: 0;">📊 داشبورد مدیریتی v13.5</h2>
                <div style="background: #1a1a1a; padding: 5px 15px; border-radius: 8px; border: 1px solid #444; color: white;">
                    BTC: <span style="color: {'#2ecc71' if btc_status == 'BULLISH' else '#e74c3c'};">{btc_status} {btc_icon}</span>
                </div>
            </div>
            <table style="width: 100%; border-collapse: collapse; color: white; text-align: center; font-size: 13px;">
                <tr style="background-color: #111; border-bottom: 2px solid #f1c40f;">
                    <th style="padding: 12px;">ارز</th><th style="padding: 12px;">تایم</th>
                    <th style="padding: 12px;">رژیم بازار</th><th style="padding: 12px;">سیگنال</th>
                    <th style="padding: 12px;">قدرت AI</th>
                </tr>
        """
        for _, row in df_dash.iterrows():
            html_table += f"""
                <tr style="border-bottom: 1px solid #222; background-color: {row['bg']};">
                    <td style="padding: 10px; font-weight: bold;">{row['ارز']}</td>
                    <td style="padding: 10px; color: #aaa;">{row['تایم‌فریم']}</td>
                    <td style="padding: 10px; font-size: 11px;">{row['رژیم']}</td>
                    <td style="padding: 10px; color: {row['color']}; font-weight: bold;">{row['وضعیت']}</td>
                    <td style="padding: 10px; font-weight: bold;">{row['قدرت']}</td>
                </tr>
            """
        html_table += "</table></div>"
        display(HTML(html_table))

    except Exception as e:
        print(f"❌ خطای داشبورد: {e}")

async def main():
    TIMEFRAMES = {"15m": 15*60, "1h": 60*60, "4h": 4*60*60, "1d": 24*60*60}
    last_run = {tf: 0 for tf in TIMEFRAMES}
    last_status_time = 0

    output.clear()
    print("🚀 سیستم Elite v13.5 با موفقیت لود شد...")
    send_telegram(f"🚀 **هسته مرکزی v13.5 آنلاین شد**\n📊 داشبورد خودکار تلگرام فعال گردید.")

    while True:
        try:
            current_time = time.time()

            # آپدیت داشبورد (هر ۵ دقیقه یکبار)
            if current_time - last_status_time >= 300:
                output.clear(wait=True)
                show_dashboard()
                gc.collect()
                if 'K' in globals(): K.clear_session()
                last_status_time = current_time

            for tf, interval in TIMEFRAMES.items():
                if current_time - last_run[tf] >= interval:
                    last_run[tf] = current_time
                    print(f"⏰ اسکن {tf} آغاز شد...")

                    for coin, sym in CRYPTOS.items():
                        try:
                            df = get_data(coin, interval=tf)
                            if df is not None and not df.empty:
                                df = add_indicators(df)
                                msg = analyze(df, sym, tf, coin)

                                # ذخیره رژیم بازار برای داشبورد
                                if "رونددار" in msg: regime_val = "TRENDING 💪"
                                elif "ساید" in msg: regime_val = "SIDEWAYS 💤"
                                elif "نوسانی" in msg: regime_val = "VOLATILE ⚡"
                                else: regime_val = "STABLE ⚖️"

                                if coin not in market_regimes: market_regimes[coin] = {}
                                market_regimes[coin][tf] = regime_val

                                if "WAIT" not in msg and "صبر" not in msg:
                                    send_telegram(msg)

                                await asyncio.sleep(1.5)
                        except Exception as e:
                            print(f"⚠️ خطا در {sym}: {e}")

                    output.clear(wait=True)
                    show_dashboard()

        except Exception as e:
            print(f"🚨 اختلال سیستمی: {e}")
            await asyncio.sleep(30)

        await asyncio.sleep(10)

if __name__ == "__main__":
    try:
        await main()
    except KeyboardInterrupt:
        print("\n🛑 توقف دستی.")
    except Exception as fatal_e:
        print(f"\n🔄 راه اندازی مجدد: {fatal_e}")
        time.sleep(10)
        asyncio.create_task(main())
