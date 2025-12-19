import streamlit as st

def apply_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;500;800&display=swap');
        
        /* کل صفحه */
        html, body, [class*="css"] {
            font-family: 'Vazirmatn', sans-serif;
            direction: rtl;
            text-align: right;
            background-color: #0e1117;
        }

        /* کارت سیگنال شیشه‌ای */
        .signal-card {
            background: linear-gradient(135deg, #1e222d 0%, #161a1e 100%);
            border: 1px solid #31353f;
            border-radius: 20px;
            padding: 25px;
            color: #e0e3eb;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            margin-bottom: 20px;
        }

        /* تیترها */
        .elite-header {
            background: linear-gradient(90deg, #f0b90b, #ffea00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 24px;
            text-align: center;
            margin-bottom: 10px;
            letter-spacing: 1px;
        }

        /* ردیف‌های داده */
        .info-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }

        .info-label { color: #848e9c; font-size: 14px; }
        .info-value { font-weight: 600; font-size: 15px; color: #ffffff; }

        /* باکس‌های تارگت */
        .target-box {
            background: rgba(0, 206, 201, 0.1);
            border: 1px solid rgba(0, 206, 201, 0.3);
            border-radius: 8px;
            padding: 8px 12px;
            margin: 5px 0;
            display: flex;
            justify-content: space-between;
        }
        
        .stop-box {
            background: rgba(255, 118, 117, 0.1);
            border: 1px solid rgba(255, 118, 117, 0.3);
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 15px;
            display: flex;
            justify-content: space-between;
        }

        /* دکمه اصلی */
        .stButton>button {
            width: 100%;
            border-radius: 12px;
            height: 55px;
            background: linear-gradient(90deg, #f0b90b 0%, #d4a306 100%);
            color: #000;
            font-weight: 800;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(240, 185, 11, 0.4);
        }
        </style>
    """, unsafe_allow_html=True)
