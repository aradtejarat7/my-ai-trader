import streamlit as st

def apply_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');
        html, body, [class*="css"] { font-family: 'Vazirmatn', sans-serif; direction: rtl; text-align: right; }
        .signal-card {
            background: #161a1e; border: 1px solid #2b2f36; border-radius: 15px;
            padding: 15px; color: white; line-height: 1.8;
        }
        .highlight { color: #f0b90b; font-weight: bold; }
        .stButton>button { width: 100%; border-radius: 10px; height: 50px; background: #f0b90b; color: black; }
        </style>
    """, unsafe_allow_html=True)
