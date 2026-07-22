"""Geospatial Analysis page: transaction/district/planning-area maps & rankings (Google
Maps), plus location/tenure/floor/sale-type/MRT $PSF premium comparisons."""
import json
import os

import duckdb
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import streamlit.components.v1 as components

from backend.data_pipeline import (load_data, type_filter, load_mrt_stations,
                                   DISTRICT_CENTROIDS, DISTRICT_LABELS)

BLUE, RED = "#2E7DF7", "#E4572E"

GMAP_TEMPLATE = """
<div id="map" style="height:500px;width:100%;border-radius:8px;"></div>
<script>
function initMap(){
  const map = new google.maps.Map(document.getElementById("map"),
    {center:{lat:1.31,lng:103.84}, zoom:11, mapTypeControl:true});
  const data = __DATA__; const tmax = __TMAX__;
  const info = new google.maps.InfoWindow();
  data.forEach(function(p){
    const c = new google.maps.Circle({
      strokeColor:p.color, strokeOpacity:0.9, strokeWeight:1,
      fillColor:p.color, fillOpacity:0.55, map:map,
      center:{lat:p.lat, lng:p.lon}, radius:300+(p.transactions/tmax)*1900});
    c.addListener("click", function(){
      info.setPosition({lat:p.lat, lng:p.lon});
      info.setContent("<b>"+p.label+"</b><br>Avg $PSF: "+Math.round(p.avg_psf)+
                      "<br>Transactions: "+p.transactions);
      info.open(map);
    });
  });
}
</script>
<script async src="https://maps.googleapis.com/maps/api/js?key=__KEY__&callback=initMap"></script>
"""

TXN_MAP_TEMPLATE = """
<div id="map" style="height:600px;width:100%;border-radius:8px;"></div>
<script>
function initMap(){
  const map = new google.maps.Map(document.getElementById("map"),
    {center:{lat:1.31,lng:103.84}, zoom:12, mapTypeControl:true});
  const txData = __TXDATA__; const mrtData = __MRTDATA__;
  const info = new google.maps.InfoWindow();
  txData.forEach(function(p){
    const approx = p.precision === "street";
    const c = new google.maps.Circle({
      strokeColor:p.color, strokeOpacity: approx ? 0.5 : 0.8, strokeWeight:1,
      fillColor:p.color, fillOpacity: approx ? 0.35 : 0.65, map:map,
      center:{lat:p.lat, lng:p.lon}, radius:40});
    c.addListener("click", function(){
      info.setPosition({lat:p.lat, lng:p.lon});
      info.setContent("<b>"+p.label+"</b><br>$PSF: "+Math.round(p.psf)+"<br>"+p.date+
                      (approx ? "<br><i>Approximate — building not in OneMap's index; "+
                                "placed on its street.</i>" : ""));
      info.open(map);
    });
  });
  mrtData.forEach(function(m){
    const marker = new google.maps.Marker({
      position:{lat:m.lat, lng:m.lon}, map:map, title:m.station,
      icon:{path: google.maps.SymbolPath.CIRCLE, scale:5, fillColor:"#00A651",
            fillOpacity:1, strokeColor:"#ffffff", strokeWeight:1.5}});
    marker.addListener("click", function(){
      info.setPosition({lat:m.lat, lng:m.lon});
      info.setContent("<b>"+m.station+"</b> MRT/LRT station");
      info.open(map);
    });
  });
}
</script>
<script async src="https://maps.googleapis.com/maps/api/js?key=__KEY__&callback=initMap"></script>
"""


def val_to_hex(v, lo, hi):
    t = 0.0 if hi == lo or pd.isna(v) else (v - lo) / (hi - lo)
    c0, c1 = (33, 48, 90), (244, 208, 63)   # dark blue -> yellow
    return "#%02x%02x%02x" % tuple(int(c0[i] + (c1[i] - c0[i]) * t) for i in range(3))


def gmaps_key():
    try:
        return st.secrets["GOOGLE_MAPS_API_KEY"]
    except Exception:
        return os.environ.get("GOOGLE_MAPS_API_KEY", "")


def render_bubble_map(d, unit_noun):
    """d needs lat/lon (bubble position), avg_psf (colour), transactions (size), label
    (tooltip). Used for both District and Planning Area aggregate views — the position
    is the mean lat/lon of that group's geocoded transactions (real building locations,
    not a hand-set centroid; §8), so it moves with wherever the group's actual deals
    are, not a fixed approximate point."""
    key = gmaps_key()
    if not key:
        st.error("Google Maps API key not set. Add `GOOGLE_MAPS_API_KEY` to "
                 "`.streamlit/secrets.toml` (local) or the app's Settings → Secrets (Cloud).")
        return
    lo, hi = d["avg_psf"].min(), d["avg_psf"].max()
    tmax = max(int(d["transactions"].max()), 1)
    pts = (d.assign(color=d["avg_psf"].map(lambda v: val_to_hex(v, lo, hi)))
           [["lat", "lon", "label", "avg_psf", "transactions", "color"]]
           .round({"avg_psf": 0}).to_dict("records"))
    html = (GMAP_TEMPLATE.replace("__KEY__", key)
            .replace("__DATA__", json.dumps(pts)).replace("__TMAX__", str(tmax)))
    components.html(html, height=520)
    st.caption(f"Each bubble is a {unit_noun}, positioned at the average location of its "
               f"geocoded transactions. Colour shows average $PSF (dark blue ${lo:,.0f} "
               f"→ yellow ${hi:,.0f}); bubble size shows the number of transactions. Click "
               "a bubble for details.")


def render_transaction_map(txf, mrt):
    """One dot per transaction at its real geocoded building location (backend/
    geocode_buildings.py) — transactions in the same building necessarily share a
    point, since that's the precision OneMap actually resolves to (§9 of README), not
    an aggregation choice made here. Green dots are MRT/LRT station exits."""
    key = gmaps_key()
    if not key:
        st.error("Google Maps API key not set. Add `GOOGLE_MAPS_API_KEY` to "
                 "`.streamlit/secrets.toml` (local) or the app's Settings → Secrets (Cloud).")
        return
    geocoded = txf.dropna(subset=["lat", "lon"])
    n_missing = len(txf) - len(geocoded)
    n_approx = int((geocoded["geocode_precision"] == "street").sum())
    lo, hi = geocoded["psf"].min(), geocoded["psf"].max()
    tx_pts = (geocoded.assign(color=geocoded["psf"].map(lambda v: val_to_hex(v, lo, hi)),
                              date=geocoded["sale_date"].dt.strftime("%Y-%m-%d"),
                              label=geocoded["Project Name"], precision=geocoded["geocode_precision"])
             [["lat", "lon", "psf", "color", "date", "label", "precision"]]
             .round({"psf": 0}).to_dict("records"))
    mrt_pts = mrt[["station", "lat", "lon"]].to_dict("records")
    html = (TXN_MAP_TEMPLATE.replace("__KEY__", key)
            .replace("__TXDATA__", json.dumps(tx_pts)).replace("__MRTDATA__", json.dumps(mrt_pts)))
    components.html(html, height=620)
    cov = len(geocoded) / len(txf) * 100 if len(txf) else 0
    st.caption(f"{len(geocoded):,} of {len(txf):,} transactions plotted ({cov:.0f}% geocoded"
               f"{f', {n_missing} at buildings not in OneMap' if n_missing else ''}"
               f"{f', {n_approx} shown faded — street-level approximate, §8' if n_approx else ''}"
               f"). Each dot is one transaction, coloured by $PSF (dark blue ${lo:,.0f} → "
               f"yellow ${hi:,.0f}). Green dots are MRT/LRT station exits.")


RANKING_VIEWPORT = 440   # px — fixed visible height shared by the bar and the table
RANKING_ROW_H = 26       # px per bar in the chart's true (scrollable) height


def ranking_bar_and_table(df, key_col, label_col, sel_key, extra_cols=None):
    """Shared "Map & Ranking" layout piece: a click-to-select ranking bar chart beside
    a summary table, on the same row, both showing EVERY row of `df` (no top-N cut) at
    the same fixed viewport height — each scrolls independently if `df` doesn't fit,
    via `st.container(height=...)` for the chart and `st.dataframe`'s own `height` for
    the table. `df` (already sorted descending by avg_psf) needs `key_col` (the
    grouping key used for selection + the caller's drill-down query), `label_col`
    (display name — may equal key_col), `avg_psf`, `median_psf`, `transactions`.
    `extra_cols` adds pre-formatted columns to the table only (e.g. a building's last
    trade date). Returns the selected `key_col` value: the clicked bar, or the
    top-ranked row if nothing's been clicked yet."""
    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        pick = alt.selection_point(fields=[key_col], on="click", empty=False, name=f"pick_{sel_key}")
        bar = alt.Chart(df).mark_bar().encode(
            y=alt.Y(f"{label_col}:N", sort="-x", title=None),
            x=alt.X("avg_psf:Q", title="Avg $PSF"),
            color=alt.Color("avg_psf:Q", scale=alt.Scale(scheme="viridis"), legend=None),
            opacity=alt.condition(pick, alt.value(1.0), alt.value(0.45)),
            tooltip=[f"{label_col}:N", alt.Tooltip("avg_psf:Q", format=",.0f"),
                     alt.Tooltip("transactions:Q", format=",.0f")]
        ).add_params(pick).properties(height=max(RANKING_VIEWPORT, RANKING_ROW_H * len(df)))
        with st.container(height=RANKING_VIEWPORT):
            event = st.altair_chart(bar, on_select="rerun", key=f"rank_{sel_key}", width="stretch")
    with col_table:
        table_cols = [label_col, "avg_psf", "median_psf", "transactions"] + (extra_cols or [])
        st.dataframe(df[table_cols].style.format(
            {"avg_psf": "{:,.0f}", "median_psf": "{:,.0f}", "transactions": "{:,.0f}"}),
            width="stretch", hide_index=True, height=RANKING_VIEWPORT)

    selected = None
    try:
        rows = event.selection.get(f"pick_{sel_key}") if event and event.selection else None
        if rows:
            selected = rows[0][key_col]
    except Exception:
        selected = None
    if selected is None:
        selected = df.iloc[0][key_col]
    return selected


def render_detail_table(txf, where_col, where_val, title, download_stem):
    """Deep-dive: every transaction where `where_col == where_val` (a parameterized
    DuckDB query — safe regardless of whether where_val is an int or a string)."""
    tdf = txf[["Project Name", "Address", "sale_date", "area_sqft", "psf", "price",
               "tenure_type", "type_of_sale", where_col]].copy()
    # ranking_bar_and_table's selection can hand back a numpy scalar (e.g. int64 for
    # district) — DuckDB's parameter binder only accepts native Python types.
    where_val = where_val.item() if hasattr(where_val, "item") else where_val
    detail = duckdb.sql(f"""
        SELECT "Project Name" AS project, Address AS address,
               strftime(sale_date, '%Y-%m-%d') AS sale_date,
               area_sqft, psf AS unit_psf, price AS transacted_price,
               tenure_type AS tenure, type_of_sale AS sale_type
        FROM tdf
        WHERE "{where_col}" = ?
        ORDER BY sale_date DESC
    """, params=[where_val]).df()
    st.markdown(f"**Transactions in {title}**  ·  {len(detail):,} records")
    st.dataframe(detail.style.format({"area_sqft": "{:,.0f}", "unit_psf": "{:,.0f}",
                                      "transacted_price": "{:,.0f}"}),
                 width="stretch", hide_index=True)
    st.download_button("Download transactions (CSV)", detail.to_csv(index=False),
                       file_name=f"transactions_{download_stem}.csv", mime="text/csv",
                       key=f"dl_{download_stem}")


def premium_bar_chart(txf, group_col, sort_order, color, x_title=None, height=300):
    """One Premium Factors comparison: median $psf by category (`group_col`, already
    a column on `txf`), with sample size in the tooltip."""
    by = (txf.groupby(group_col, observed=True)
          .agg(median_psf=("psf", "median"), transactions=("psf", "size")).reset_index())
    st.altair_chart(alt.Chart(by).mark_bar(color=color).encode(
        x=alt.X(f"{group_col}:N", title=x_title, sort=sort_order),
        y=alt.Y("median_psf:Q", title="Median $PSF", scale=alt.Scale(zero=False)),
        tooltip=[f"{group_col}:N", alt.Tooltip("median_psf:Q", format=",.0f"),
                 alt.Tooltip("transactions:Q", format=",.0f")]
    ).properties(height=height), width="stretch")


st.title("Geospatial Analysis")
st.caption("How location affects office pricing — transaction, district, and planning-area "
           "maps & rankings, plus location/tenure/floor/MRT price premiums.")

tx, _ = load_data()
tx, view_choice = type_filter(tx)
if view_choice != "Strata":
    st.warning(f"Transaction Type = **{view_choice}**: Land deals price $PSF on land/site "
               "area, not unit area, so the map/rankings below mix or show a different basis "
               "than Strata-only $PSF.", icon="⚠️")
if tx.empty:
    st.warning("No transactions for this Transaction Type selection.")
    st.stop()

projects_all = sorted(tx["Project Name"].dropna().unique())
streets_all = sorted(tx["street"].dropna().unique())

c = st.columns(3)
yrs = sorted(tx["year"].unique())
yr_range = c[0].select_slider("Year range", options=yrs, value=(yrs[0], yrs[-1]))
tos_sel = c[1].multiselect("Type of sale (empty = all)", ["Resale", "New Sale", "Sub Sale"])
ten_sel = c[2].multiselect("Tenure (empty = all)", ["Leasehold", "Freehold"])
c2 = st.columns(2)
name_sel = c2[0].multiselect("Project name (empty = all)", projects_all)
street_sel = c2[1].multiselect("Street (empty = all)", streets_all)

pick = lambda sel, allv: allv if not sel else sel
txf = tx[tx["year"].between(yr_range[0], yr_range[1])
         & tx["type_of_sale"].isin(pick(tos_sel, ["Resale", "New Sale", "Sub Sale"]))
         & tx["tenure_type"].isin(pick(ten_sel, ["Leasehold", "Freehold"]))
         & tx["Project Name"].isin(pick(name_sel, projects_all))
         & tx["street"].isin(pick(street_sel, streets_all))].copy()

if txf.empty:
    st.warning("No transactions match the current filters. Widen the selection above.")
    st.stop()
st.caption(f"{len(txf):,} {view_choice} transactions · {yr_range[0]}–{yr_range[1]}")

t1, t2, t3, t4 = st.tabs(["Transaction Map & Ranking", "District Map & Ranking",
                          "Planning Area Map & Ranking", "Premium Factors"])

# ---------------------------------------------------------------- individual transactions + building ranking
with t1:
    st.markdown("**Every transaction, at its real building location**")
    render_transaction_map(txf, load_mrt_stations())

    st.divider()
    bldg = (txf.groupby("Project Name")
            .agg(avg_psf=("psf", "mean"), median_psf=("psf", "median"),
                 transactions=("psf", "size"), last_trade=("sale_date", "max")).reset_index())
    n_bldg_all = len(bldg)
    bldg = bldg[bldg["transactions"] >= 3].sort_values("avg_psf", ascending=False)
    bldg["last_trade"] = bldg["last_trade"].dt.strftime("%Y-%m-%d")
    st.markdown("**Building ranking** · click a bar to see that building's transactions")
    st.caption(f"{len(bldg):,} of {n_bldg_all:,} buildings shown — fewer than 3 transactions "
               "excluded (too few for a reliable average).")
    if bldg.empty:
        st.info("No buildings with at least 3 transactions in the current filter.")
    else:
        sel_bldg = ranking_bar_and_table(bldg, "Project Name", "Project Name", "bldg",
                                         extra_cols=["last_trade"])
        render_detail_table(txf, "Project Name", sel_bldg, title=sel_bldg,
                            download_stem=sel_bldg.replace(" ", "_"))

# ---------------------------------------------------------------- district map + ranking
with t2:
    d = (txf.dropna(subset=["postal_district"]).groupby("postal_district")
         .agg(avg_psf=("psf", "mean"), median_psf=("psf", "median"),
              transactions=("psf", "size"), avg_price=("price", "mean"),
              lat=("lat", "mean"), lon=("lon", "mean")).reset_index())
    d["district"] = d["postal_district"].astype(int)
    d["label"] = d["district"].map(DISTRICT_LABELS)
    # bubble position = mean of that district's real geocoded transactions; only
    # fall back to the hand-set approximate centroid (DISTRICT_CENTROIDS) if a
    # district has zero geocoded transactions in the current filter.
    d["lat"] = d["lat"].fillna(d["district"].map(lambda x: DISTRICT_CENTROIDS.get(x, (None, None))[0]))
    d["lon"] = d["lon"].fillna(d["district"].map(lambda x: DISTRICT_CENTROIDS.get(x, (None, None))[1]))
    d = d.dropna(subset=["lat", "lon"]).sort_values("avg_psf", ascending=False)

    st.markdown("**Average office $PSF by district**")
    render_bubble_map(d, "district")

    st.divider()
    st.markdown("**District ranking** · click a bar to see that district's transactions")
    sel_dist = ranking_bar_and_table(d, "district", "label", "dist")
    render_detail_table(txf, "postal_district", sel_dist,
                        title=DISTRICT_LABELS.get(sel_dist, sel_dist), download_stem=f"D{sel_dist:02d}")

# ---------------------------------------------------------------- planning area map + ranking
with t3:
    pa = (txf.groupby("sub_market").agg(avg_psf=("psf", "mean"), median_psf=("psf", "median"),
                                        transactions=("psf", "size"),
                                        lat=("lat", "mean"), lon=("lon", "mean")).reset_index())
    pa = pa[pa["transactions"] >= 10].sort_values("avg_psf", ascending=False)

    st.markdown("**Average office $PSF by planning area**")
    pa_map = pa.assign(label=pa["sub_market"]).dropna(subset=["lat", "lon"])
    if pa_map.empty:
        st.info("No geocoded transactions to plot for the current filter.")
    else:
        render_bubble_map(pa_map, "planning area")

    st.divider()
    st.markdown("**Planning area ranking** (≥10 transactions) · click a bar to see that area's "
               "transactions")
    st.caption("Areas with fewer than 10 transactions are excluded — too few for a reliable average.")
    if pa.empty:
        st.info("No planning areas with at least 10 transactions in the current filter.")
    else:
        sel_pa = ranking_bar_and_table(pa, "sub_market", "sub_market", "pa")
        render_detail_table(txf, "sub_market", sel_pa, title=sel_pa,
                            download_stem=sel_pa.replace(" ", "_"))

# ---------------------------------------------------------------- premium factors
with t4:
    row1a, row1b = st.columns(2)
    with row1a:
        st.markdown("**Location premium** — Downtown Core vs. rest of market")
        cbd = txf[txf["sub_market"] == "Downtown Core"]["psf"].median()
        rest = txf[txf["sub_market"] != "Downtown Core"]["psf"].median()
        mc = st.columns(2)
        mc[0].metric("Downtown Core median $PSF", f"{cbd:,.0f}")
        mc[1].metric("Rest of market median $PSF", f"{rest:,.0f}",
                     f"{(cbd/rest-1)*100:+.0f}% CBD premium" if rest else None)
        loc_txf = txf.assign(location=np.where(txf["sub_market"] == "Downtown Core",
                                               "Downtown Core", "Rest of market"))
        premium_bar_chart(loc_txf, "location", ["Downtown Core", "Rest of market"], BLUE)
    with row1b:
        st.markdown("**Tenure premium** — Freehold vs. Leasehold")
        fh = txf[txf["tenure_type"] == "Freehold"]["psf"].median()
        lh = txf[txf["tenure_type"] == "Leasehold"]["psf"].median()
        mc = st.columns(2)
        mc[0].metric("Freehold median $PSF", f"{fh:,.0f}")
        mc[1].metric("Leasehold median $PSF", f"{lh:,.0f}",
                     f"{(fh/lh-1)*100:+.0f}% freehold premium" if lh else None)
        premium_bar_chart(txf, "tenure_type", ["Freehold", "Leasehold"], RED)

    st.divider()
    row2a, row2b = st.columns(2)
    with row2a:
        st.markdown("**Floor-level premium**")
        floor_bins = [0, 5, 15, 100]
        floor_labels = ["Low (1–5F)", "Mid (6–15F)", "High (16F+)"]
        floor_txf = txf.assign(floor_tier=pd.cut(txf["floor"], floor_bins, labels=floor_labels))
        premium_bar_chart(floor_txf, "floor_tier", floor_labels, BLUE)
    with row2b:
        st.markdown("**Sale-type premium**")
        premium_bar_chart(txf, "type_of_sale", ["New Sale", "Resale", "Sub Sale"], RED)

    st.divider()
    row3a, row3b = st.columns(2)
    with row3a:
        st.markdown("**Unit-size premium**")
        size_labels = ["<=500", "500-1k", "1k-2k", "2k-5k", ">5k"]
        premium_bar_chart(txf, "size_band", size_labels, BLUE)
    with row3b:
        st.markdown("**Deal-size premium**")
        deal_labels = ["<$5M", "$5-10M", ">$10M"]
        premium_bar_chart(txf, "deal_band", deal_labels, RED)

    st.divider()
    row4a, row4b = st.columns(2)
    with row4a:
        st.markdown("**MRT density (400m) premium**")
        count_txf = txf.dropna(subset=["lat", "lon"])
        if count_txf.empty:
            st.info("No geocoded transactions in the current filter.")
        else:
            st.caption("Number of distinct stations within a 5-min (400m) walk — a clean, "
                       "roughly monotonic premium in the current data.")
            premium_bar_chart(count_txf, "mrt_count_400m", [0, 1, 2, 3], BLUE)
    with row4b:
        st.markdown("**Interchange-access (400m) premium**")
        inter_txf = txf.dropna(subset=["lat", "lon"])
        if inter_txf.empty:
            st.info("No geocoded transactions in the current filter.")
        else:
            st.caption("Whether an interchange station (≥2 MRT/LRT lines, verified against "
                       "LTA's official station-line list, §8) is within 400m — weaker than "
                       "MRT density: the interchange effect is largely already captured by "
                       "station density and CBD location, so this adds little on its own.")
            premium_bar_chart(inter_txf, "near_interchange_400m", [False, True], RED)

    st.divider()
    st.markdown("**MRT-accessibility premium**")
    mrt_txf = txf.dropna(subset=["dist_to_mrt_km"])
    if len(mrt_txf) < 10:
        st.info("Not enough geocoded transactions in the current filter to show an "
                "MRT-accessibility breakdown.")
    else:
        bands = [0, 0.2, 0.4, 0.6, 0.8, 1.0, np.inf]
        band_labels = ["≤200m", "200–400m", "400–600m", "600–800m", "800m–1km", ">1km"]
        mrt_txf = mrt_txf.assign(mrt_band=pd.cut(mrt_txf["dist_to_mrt_km"], bands, labels=band_labels))
        corr = mrt_txf[["dist_to_mrt_km", "psf"]].corr().iloc[0, 1]
        st.caption(f"{len(mrt_txf):,} geocoded transactions, straight-line distance to the "
                   f"nearest MRT/LRT exit. Correlation (distance vs $PSF): **{corr:+.2f}** "
                   f"(negative = pricier closer to a station, as expected).")
        premium_bar_chart(mrt_txf, "mrt_band", band_labels, BLUE,
                          x_title="Distance to nearest MRT/LRT", height=320)
