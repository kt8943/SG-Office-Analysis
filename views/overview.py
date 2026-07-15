"""Overview / landing page — what the app does, headline numbers, and how to use it."""
import streamlit as st

from data_pipeline import load_data

st.title("Singapore Office Price Analysis")
st.markdown(
    "An interactive look at **Singapore strata office sale transactions (2010–2026)**, "
    "enriched with market and macro indicators. Use the sidebar to move between pages."
)

tx, _ = load_data()
c = st.columns(4)
c[0].metric("Transactions", f"{len(tx):,}")
c[1].metric("Period", f"{tx['year'].min()}–{tx['year'].max()}")
c[2].metric("Planning areas", f"{tx['sub_market'].nunique()}")
c[3].metric("Median $PSF", f"{tx['psf'].median():,.0f}")

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("What this app does")
    st.markdown(
        "- Tracks how office **prices, $PSF, and volume** have moved over time\n"
        "- Compares trends against **macro factors** (CPI, interest rates, GDP, rent & price indices)\n"
        "- Shows **where** value sits across Singapore **districts and planning areas**\n"
        "- Lets you **filter** by year, planning area, tenure, floor, size and sale type"
    )
    st.subheader("Pages")
    st.markdown(
        "- **Trends (2010–2026)** — price/$PSF/volume over time, seasonality, macro factors\n"
        "- **Geospatial Analysis** — district $PSF map, district & planning-area rankings, "
        "CBD & tenure premiums"
    )

with right:
    st.subheader("Data & method")
    st.markdown(
        "- **Source:** URA commercial (office) transactions + SingStat/MAS market & macro series\n"
        "- **Scope:** strata office units only; whole-building/en-bloc deals excluded\n"
        "- **Focus:** Central Region (≈92% of transactions); market series use the Central geography\n"
        "- **$PSF** = transacted unit price per sq ft; **real $PSF** is CPI-adjusted (2024 base)"
    )
    st.info(
        "The most recent quarter may show blanks for GDP/CPI because those official releases "
        "lag transactions. Correlations shown are descriptive, not causal."
    )
