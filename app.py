"""
Singapore Office Price — Trends (Phase 1 frontend)

Top filter bar -> KPI cards -> tabbed views (Avg Price / PSF / Transaction Volume /
Seasonality / Macro Factors) with event annotations, a seasonality heatmap, nested
macro sub-tabs, and collapsible AI-style insights. Data table at the bottom.

Run:  streamlit run app.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

alt.data_transformers.disable_max_rows()
DATA = Path(__file__).parent / "Data"
st.set_page_config(page_title="SG Office Price Trends", layout="wide")
BLUE, RED = "#2E7DF7", "#E4572E"

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

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# objective, dated macro shocks only (annotated on the PSF chart, not mix-driven avg price)
EVENTS = {2020: "COVID", 2022: "Rate-Hike Surge"}
MACRO_TABS = {"PSF vs CPI": ("cpi", "CPI (index, 2024=100)"),
              "PSF vs Interest Rate": ("sora_3m", "SORA 3M (%)"),
              "PSF vs GDP": ("gdp_growth", "GDP growth (YoY %)"),
              "PSF vs Rent Index": ("rent_index", "URA Office Rent Index"),
              "PSF vs Office Price Index": ("price_index", "URA Office Price Index")}


# --------------------------------------------------------------- data loading
def _load_market(path, col, name):
    df = pd.read_csv(DATA / path)
    df["quarter"] = pd.PeriodIndex(df["Quarter"], freq="Q")
    df[name] = pd.to_numeric(df[col], errors="coerce")
    return df[["quarter", name]]


def _load_dos_wide(path, series, name):
    raw = pd.read_csv(DATA / path, header=None, dtype=str)
    hdr = raw.index[raw[0].astype(str).str.strip() == "Data Series"][0]
    df = pd.read_csv(DATA / path, skiprows=hdr)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "series"})
    row = df[df["series"].astype(str).str.strip() == series]
    rec = []
    for col, val in row.iloc[0, 1:].items():
        m = re.match(r"(\d{4})\s*([1-4])Q", str(col).strip())
        if m:
            rec.append((pd.Period(f"{m.group(1)}Q{m.group(2)}", freq="Q"),
                        pd.to_numeric(val, errors="coerce")))
    return pd.DataFrame(rec, columns=["quarter", name]).dropna()


@st.cache_data
def load_data():
    tx = pd.read_csv(DATA / "CommercialTransaction_byProject.csv", thousands=",")
    tx["sale_date"] = pd.to_datetime(tx["Sale Date"], format="%d %b %Y")
    tx = tx[tx["Type of Area"] == "Strata"].copy()
    enbloc = ((tx["Area (SQFT)"] > 30000) | (tx["Transacted Price ($)"] > 50_000_000)
              | (tx["Address"].str.contains("ENBLOC", case=False, na=False))
              | (tx["Unit Price ($ PSF)"] > 10000))
    tx = tx[~enbloc].copy()

    tx["price"] = tx["Transacted Price ($)"]
    tx["psf"] = tx["Unit Price ($ PSF)"]
    tx["area_sqft"] = tx["Area (SQFT)"]
    tx["quarter"] = tx["sale_date"].dt.to_period("Q")
    tx["year"] = tx["sale_date"].dt.year
    tx["month"] = tx["sale_date"].dt.month
    tx["quarter_of_year"] = tx["sale_date"].dt.quarter
    tx["sub_market"] = tx["Planning Area"]
    tx["tenure_type"] = np.where(
        tx["Tenure"].str.contains("Freehold|999", case=False, na=False), "Freehold", "Leasehold")
    tx["size_band"] = pd.cut(tx["Area (SQFT)"], [0, 500, 1000, 2000, 5000, np.inf],
                             labels=["<=500", "500-1k", "1k-2k", "2k-5k", ">5k"])
    tx["type_of_sale"] = tx["Type of Sale"]
    tx["floor"] = tx["Address"].str.extract(r"#(\d+)-")[0].astype(float)
    tx["floor"] = tx["floor"].fillna(tx["floor"].median()).astype(int)

    market = _load_market("Property Price Index of Office Space.csv",
                          "Property Price Index of Office Space in Central Region (INDEX)", "price_index")
    for path, col, name in [
        ("Rental Index of Private Sector Office Space.csv",
         "Rental Index of Private Sector Office Space in Central Region (INDEX)", "rent_index"),
        ("Vacancy Rate of Private Sector Office Space.csv",
         "Vacancy Rate of Private Sector Office Space in Central Region (per cent)", "vacancy_rate"),
        ("Private Sector Office Space under Construction:Pipeline:Planned Supply.csv",
         "Supply of Private Sector Office Space in the Pipeline ('000 SQ M GROSS)", "supply_pipeline"),
    ]:
        market = market.merge(_load_market(path, col, name), on="quarter", how="outer")
    for path, series, name in [("GDP Growth Rate.csv", "GDP At Current Market Prices", "gdp_growth"),
                               ("CPI quarterly.csv", "All Items", "cpi")]:
        market = market.merge(_load_dos_wide(path, series, name), on="quarter", how="outer")

    sora = pd.read_csv(DATA / "Domestic Interest Rates (9).csv", skiprows=6)
    sora.columns = [str(c).strip() for c in sora.columns]
    sora["date"] = pd.to_datetime(sora["SORA Publication Date"], format="%d %b %Y", errors="coerce")
    sora["sora_3m"] = pd.to_numeric(sora["Compound SORA - 3 month"], errors="coerce")
    sora = sora.dropna(subset=["date", "sora_3m"])
    sq = sora.groupby(sora["date"].dt.to_period("Q"))["sora_3m"].mean().reset_index()
    sq.columns = ["quarter", "sora_3m"]
    market = market.merge(sq, on="quarter", how="outer")

    un = pd.read_csv(DATA / "quarterly overall unemployment rate.csv")
    un = un[un["residential_status"] == "overall"].copy()
    un["quarter"] = pd.PeriodIndex(pd.to_datetime(un["month"]), freq="Q")
    un["unemployment"] = pd.to_numeric(un["seasonally_adjusted_unemployment_rate"], errors="coerce")
    market = market.merge(un[["quarter", "unemployment"]], on="quarter", how="outer")

    tx = tx.merge(market[["quarter", "cpi"]], on="quarter", how="left")
    tx["real_psf"] = tx["psf"] * 100 / tx["cpi"]
    return tx, market.sort_values("quarter")


def aggregate(tx, market, gran):
    key = "quarter" if gran == "Quarter" else "year"
    agg = (tx.groupby(key).agg(
        volume=("psf", "size"), total_area=("area_sqft", "sum"), avg_price=("price", "mean"),
        median_psf=("psf", "median"), mean_psf=("psf", "mean"),
        median_real_psf=("real_psf", "median"), total_value=("price", "sum")).reset_index())
    if gran == "Quarter":
        agg = agg.merge(market, on="quarter", how="left")
        agg["period_date"] = agg["quarter"].dt.to_timestamp()
        agg["period"] = agg["quarter"].astype(str)
    else:
        my = market.copy(); my["year"] = my["quarter"].dt.year
        my = my.drop(columns="quarter").groupby("year").mean(numeric_only=True).reset_index()
        agg = agg.merge(my, on="year", how="left")
        agg["period_date"] = pd.to_datetime(agg["year"].astype(str) + "-01-01")
        agg["period"] = agg["year"].astype(str)
    return agg.sort_values("period_date")


# --------------------------------------------------------------- chart helpers
def event_layer(agg):
    yr0, yr1 = agg["period_date"].dt.year.min(), agg["period_date"].dt.year.max()
    ev = pd.DataFrame([(y, l) for y, l in EVENTS.items() if yr0 <= y <= yr1],
                      columns=["year", "label"])
    if ev.empty:
        return None
    ev["period_date"] = pd.to_datetime(ev["year"].astype(str) + "-06-01")
    rule = alt.Chart(ev).mark_rule(strokeDash=[3, 3], color="#999").encode(x="period_date:T")
    txt = alt.Chart(ev).mark_text(angle=270, align="left", dx=4, dy=-6, fontSize=10,
                                  color="#777").encode(x="period_date:T", text="label:N")
    return rule + txt


def line_chart(agg, col, title, brk_df=None, events=False):
    st.markdown(f"**{title}**")
    if brk_df is not None:
        ch = alt.Chart(brk_df).mark_line(point=True).encode(
            x=alt.X("period_date:T", title=None),
            y=alt.Y(f"{col}:Q", title=title, scale=alt.Scale(zero=False)),
            color=alt.Color("sub_market:N", title="Planning Area"),
            tooltip=["period_date:T", "sub_market:N", alt.Tooltip(f"{col}:Q", format=",.0f")])
    else:
        ch = alt.Chart(agg).mark_line(point=True, color=BLUE).encode(
            x=alt.X("period_date:T", title=None),
            y=alt.Y(f"{col}:Q", title=title, scale=alt.Scale(zero=False)),
            tooltip=["period:N", alt.Tooltip(f"{col}:Q", format=",.0f")])
    ev = event_layer(agg) if (events and brk_df is None) else None
    chart = alt.layer(ch, ev) if ev is not None else ch
    st.altair_chart(chart.properties(height=380).interactive(), width="stretch")


def breakdown(txf, gran, tx_col, func, out_col):
    key = "quarter" if gran == "Quarter" else "year"
    top = txf["sub_market"].value_counts().head(6).index
    d = (txf[txf["sub_market"].isin(top)].groupby([key, "sub_market"])[tx_col]
         .agg(func).reset_index().rename(columns={tx_col: out_col}))
    d["period_date"] = (d["quarter"].dt.to_timestamp() if gran == "Quarter"
                        else pd.to_datetime(d["year"].astype(str) + "-01-01"))
    return d


def insight(agg, col, label, money=True):
    s = agg.dropna(subset=[col])
    if len(s) < 2:
        return "Not enough data in the current filter."
    f = lambda v: (f"${v:,.0f}" if money else f"{v:,.1f}")
    a, b = s.iloc[0], s.iloc[-1]
    chg = (b[col] / a[col] - 1) * 100
    pk, tr = s.loc[s[col].idxmax()], s.loc[s[col].idxmin()]
    yoy = (b[col] / s.iloc[-2][col] - 1) * 100
    return (f"**{label}** moved from {f(a[col])} in {a['period']} to {f(b[col])} in {b['period']} "
            f"(**{chg:+.1f}%** over the window). Peak {f(pk[col])} in **{pk['period']}**, "
            f"trough {f(tr[col])} in **{tr['period']}**. Most recent step: **{yoy:+.1f}%**.")


# --------------------------------------------------------------- filters (top bar)
st.title("🏢 Office Price Trends")
tx, market = load_data()
areas_all = tx["sub_market"].value_counts().index.tolist()

r1 = st.columns(3)
yrs = sorted(tx["year"].unique())
yr_range = r1[0].select_slider("Year range", options=yrs, value=(yrs[0], yrs[-1]))
sub_sel = r1[1].multiselect("Planning Area (empty = all)", areas_all)
tos_sel = r1[2].multiselect("Type of sale (empty = all)", ["Resale", "New Sale", "Sub Sale"])
r2 = st.columns(3)
fmin, fmax = int(tx["floor"].min()), int(tx["floor"].max())
floor_range = r2[0].select_slider("Floor range", options=list(range(fmin, fmax + 1)),
                                  value=(fmin, fmax))
ten_sel = r2[1].multiselect("Tenure (empty = all)", ["Leasehold", "Freehold"])
size_sel = r2[2].multiselect("Size band (empty = all)", ["<=500", "500-1k", "1k-2k", "2k-5k", ">5k"])

gran = st.radio("Granularity", ["Year", "Quarter"], horizontal=True)
brk = st.checkbox("Break down by Planning Area (top 6)")

pick = lambda sel, allv: allv if not sel else sel
txf = tx[tx["sub_market"].isin(pick(sub_sel, areas_all))
         & tx["type_of_sale"].isin(pick(tos_sel, ["Resale", "New Sale", "Sub Sale"]))
         & tx["tenure_type"].isin(pick(ten_sel, ["Leasehold", "Freehold"]))
         & tx["size_band"].astype(str).isin(pick(size_sel, ["<=500", "500-1k", "1k-2k", "2k-5k", ">5k"]))
         & tx["floor"].between(floor_range[0], floor_range[1])
         & tx["year"].between(yr_range[0], yr_range[1])].copy()

if txf.empty:
    st.warning("No transactions match the current filters. Widen the selection above.")
    st.stop()
agg = aggregate(txf, market, gran)

# --------------------------------------------------------------- KPI cards
st.caption(f"{len(txf):,} transactions after filters · {yr_range[0]}–{yr_range[1]} · en-bloc excluded")
last, prev = agg.iloc[-1], (agg.iloc[-2] if len(agg) > 1 else agg.iloc[-1])
dl = lambda c: (None if prev[c] in (0, None) or pd.isna(prev[c]) else f"{(last[c]/prev[c]-1)*100:+.1f}%")
k = st.columns(4)
k[0].metric(f"Median $PSF ({last['period']})", f"{last['median_psf']:,.0f}", dl("median_psf"))
k[1].metric("Avg transacted price", f"${last['avg_price']:,.0f}", dl("avg_price"))
k[2].metric("Volume", f"{int(last['volume']):,}", dl("volume"))
k[3].metric("Total value", f"${last['total_value']/1e6:,.0f}M", dl("total_value"))

# --------------------------------------------------------------- tabs
t1, t2, t3, t4, t5 = st.tabs(["Avg Price", "PSF", "Transaction Volume", "Seasonality", "Macro Factors"])

with t1:
    bd = breakdown(txf, gran, "price", "mean", "avg_price") if brk else None
    line_chart(agg, "avg_price", "Average Transacted Price Over Time", bd)
    with st.expander("💡 AI Insight"):
        st.markdown(insight(agg, "avg_price", "Average transacted price"))

with t2:
    bd = breakdown(txf, gran, "psf", "median", "median_psf") if brk else None
    line_chart(agg, "median_psf", "Median $PSF Over Time", bd, events=True)
    with st.expander("💡 AI Insight"):
        st.markdown(insight(agg, "median_psf", "Median $PSF"))

with t3:
    VOL = {"Number of transactions": ("volume", "Transactions", ",.0f", False),
           "Total area transacted (sqft)": ("total_area", "Area (sqft)", ",.0f", False),
           "Total value transacted ($)": ("total_value", "Value ($)", "$,.0f", True)}
    measure = st.radio("Volume measure", list(VOL), horizontal=True)
    vcol, vtitle, vfmt, vmoney = VOL[measure]
    st.markdown(f"**{measure} by {gran}**")
    st.altair_chart(alt.Chart(agg).mark_bar(color=BLUE, cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3).encode(
        x=alt.X("period:N", title=None, sort=list(agg["period"]), axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{vcol}:Q", title=vtitle),
        tooltip=["period:N", alt.Tooltip(f"{vcol}:Q", format=vfmt)]).properties(height=380),
        width="stretch")
    with st.expander("💡 AI Insight"):
        st.markdown(insight(agg, vcol, measure, money=vmoney))

with t4:
    st.markdown("Monthly patterns — identify peak and off-peak months.")
    heat = txf.groupby(["year", "month"])["psf"].median().reset_index()
    heat["month_name"] = heat["month"].map(lambda m: MONTHS[m - 1])
    st.markdown("**Median $PSF Heatmap (year × month)**")
    st.altair_chart(alt.Chart(heat).mark_rect().encode(
        x=alt.X("month_name:O", sort=MONTHS, title="Month"),
        y=alt.Y("year:O", sort="descending", title="Year"),
        color=alt.Color("psf:Q", title="Median $PSF", scale=alt.Scale(scheme="yelloworangered")),
        tooltip=["year:O", "month_name:O", alt.Tooltip("psf:Q", format=",.0f")]
    ).properties(height=360), width="stretch")

    bym = txf.groupby("month")["psf"].median().reset_index()
    bym["month_name"] = bym["month"].map(lambda m: MONTHS[m - 1])
    bym["peak"] = bym["psf"] >= bym["psf"].quantile(0.75)
    st.markdown("**Median $PSF by Month (peak months highlighted)**")
    st.altair_chart(alt.Chart(bym).mark_bar().encode(
        x=alt.X("month_name:O", sort=MONTHS, title="Month"),
        y=alt.Y("psf:Q", title="Median $PSF", scale=alt.Scale(zero=False)),
        color=alt.condition(alt.datum.peak, alt.value(RED), alt.value(BLUE)),
        tooltip=["month_name:O", alt.Tooltip("psf:Q", format=",.0f")]
    ).properties(height=300), width="stretch")

with t5:
    st.markdown("### Macro-Economic Indicators vs Office Market")
    subtabs = st.tabs(list(MACRO_TABS))
    for st_tab, (name, (col, label)) in zip(subtabs, MACRO_TABS.items()):
        with st_tab:
            corr = agg[["median_psf", col]].corr().iloc[0, 1]
            st.markdown(f"**Median $PSF vs {label}**  ·  correlation **{corr:+.2f}**")
            base = alt.Chart(agg).encode(x=alt.X("period_date:T", title=None))
            lp = base.mark_line(point=True, color=BLUE).encode(
                y=alt.Y("median_psf:Q", title="Median $PSF",
                        axis=alt.Axis(titleColor=BLUE), scale=alt.Scale(zero=False)))
            lm = base.mark_line(point=True, color=RED, strokeDash=[4, 3]).encode(
                y=alt.Y(f"{col}:Q", title=label, axis=alt.Axis(titleColor=RED),
                        scale=alt.Scale(zero=False)))
            st.altair_chart(alt.layer(lp, lm).resolve_scale(y="independent").properties(height=380),
                            width="stretch")
            with st.expander("💡 AI Insight"):
                direction = "moves with" if corr > 0.2 else "moves against" if corr < -0.2 else "shows little linear link to"
                st.markdown(f"Over {agg['period'].iloc[0]}–{agg['period'].iloc[-1]}, median $PSF "
                            f"**{direction}** {label} (correlation {corr:+.2f}). "
                            "Correlation is not causation; lead/lag effects need the modeling phase.")

# --------------------------------------------------------------- table (bottom)
st.subheader("Data table")
show = agg[["period", "volume", "total_area", "avg_price", "median_psf", "mean_psf",
            "median_real_psf", "total_value", "price_index", "rent_index", "vacancy_rate",
            "supply_pipeline", "gdp_growth", "cpi", "sora_3m", "unemployment"]].copy()
st.dataframe(show.style.format({
    "total_area": "{:,.0f}", "avg_price": "{:,.0f}", "median_psf": "{:,.0f}",
    "mean_psf": "{:,.0f}", "median_real_psf": "{:,.0f}", "total_value": "{:,.0f}",
    "price_index": "{:.1f}",
    "rent_index": "{:.1f}", "vacancy_rate": "{:.1f}", "supply_pipeline": "{:.0f}",
    "gdp_growth": "{:.1f}", "cpi": "{:.1f}", "sora_3m": "{:.2f}", "unemployment": "{:.1f}"}),
    width="stretch", hide_index=True)
st.download_button("Download table (CSV)", show.to_csv(index=False),
                   file_name=f"office_price_trend_{gran.lower()}.csv", mime="text/csv")
