import streamlit as st

def apply_mobile_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');
        html, body, [class*="css"] { font-family: 'Vazirmatn', sans-serif; direction: rtl; }
        .stMetric { background: #1e222d; border-radius: 12px; padding: 15px; border: 1px solid #31353f; }
        .radar-box { background: #11141c; border-radius: 10px; padding: 10px; border-right: 5px solid #00cec9; margin: 5px 0; }
        </style>
    """, unsafe_allow_html=True)
