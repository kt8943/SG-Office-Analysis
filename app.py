"""
SG Office Analysis — multipage app.

Entry/router: sets global page config + tab styling, then dispatches to the
Price Trends and Geographic Analysis pages (frontend/) via st.navigation.
Shared data logic lives in backend/data_pipeline.py.

Run:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="SG Office Analysis", layout="wide")

st.markdown("""
<style>
/* pill-style, high-contrast tabs */
.stTabs [data-baseweb="tab-list"] { gap: .5rem; border-bottom: none; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] {
    background: #EEF2F8; border: 1px solid #DCE3EF; border-radius: 10px;
    padding: 10px 20px; font-weight: 600; font-size: 1rem; color: #2B3648;
}
.stTabs [data-baseweb="tab"]:hover { background: #E0E8F5; color: #17335f; }
.stTabs [aria-selected="true"] {
    background: #2E7DF7 !important; color: #fff !important; border-color: #2E7DF7 !important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
</style>
""", unsafe_allow_html=True)

nav = st.navigation([
    st.Page("frontend/overview.py", title="Overview", default=True),
    st.Page("frontend/trends.py", title="Trends (2010 - 2026)"),
    st.Page("frontend/geographic.py", title="Geospatial Analysis"),
])
nav.run()
