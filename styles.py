import streamlit as st

def apply_mobile_styles():
    st.markdown("""
        <style>
        .signal-card {
            background: #161a1e;
            border: 1px solid #2b2f36;
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            line-height: 1.6;
        }
        .header-text { color: #f0b90b; font-weight: bold; font-size: 20px; text-align: center; }
        .divider { border-top: 1px solid #2b2f36; margin: 10px 0; }
        .data-row { display: flex; justify-content: space-between; margin: 5px 0; font-size: 14px; }
        .label { color: #848e9c; }
        .value { color: #ffffff; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)
