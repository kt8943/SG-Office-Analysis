"""Geospatial Analysis page: district $PSF map (Google Maps), district & planning-area rankings, CBD/tenure premium."""
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


st.title("Geospatial Analysis")
st.caption("How location affects office pricing — district $PSF map, district & planning-area "
           "rankings, and CBD / tenure premiums.")

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

t1, t2, t3 = st.tabs(["District Map & Rankings", "Planning Area Rankings", "CBD & Tenure Premium"])

# ---------------------------------------------------------------- district map + ranking
with t1:
    map_mode = st.radio("Map detail", ["Individual transactions", "District (aggregate)"],
                        horizontal=True, key="map_mode",
                        help="Individual transactions: one dot per deal at its real geocoded "
                             "building location, plus MRT/LRT stations. District (aggregate): "
                             "one bubble per postal district (28 max), for a market-wide view.")

    if map_mode == "Individual transactions":
        st.markdown("**Every transaction, at its real building location**")
        render_transaction_map(txf, load_mrt_stations())
    else:
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
        d = d.dropna(subset=["lat", "lon"])

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Average office $PSF by district**")
            render_bubble_map(d, "district")
        with right:
            st.markdown("**Avg $PSF ranking** — click a bar to see that district's transactions")
            pick = alt.selection_point(fields=["district"], on="click", empty=False, name="pick")
            rank = alt.Chart(d).mark_bar().encode(
                y=alt.Y("label:N", sort="-x", title=None),
                x=alt.X("avg_psf:Q", title="Avg $PSF"),
                color=alt.Color("avg_psf:Q", scale=alt.Scale(scheme="viridis"), legend=None),
                opacity=alt.condition(pick, alt.value(1.0), alt.value(0.45)),
                tooltip=["label:N", alt.Tooltip("avg_psf:Q", format=",.0f"),
                         alt.Tooltip("transactions:Q", format=",.0f")]
            ).add_params(pick).properties(height=520)
            event = st.altair_chart(rank, on_select="rerun", key="rank", width="stretch")

        # which district is selected? (clicked bar, else the top-ranked district)
        sel_dist = None
        try:
            rows = event.selection.get("pick") if event and event.selection else None
            if rows:
                sel_dist = int(rows[0]["district"])
        except Exception:
            sel_dist = None
        if sel_dist is None:
            sel_dist = int(d.sort_values("avg_psf", ascending=False).iloc[0]["district"])

        # transaction-level detail for the selected district — queried with DuckDB (SQL)
        tdf = txf[["Project Name", "Address", "sale_date", "area_sqft", "psf", "price",
                   "tenure_type", "type_of_sale", "postal_district"]].copy()
        tdf["postal_district"] = tdf["postal_district"].astype("Int64")
        detail = duckdb.sql(f"""
            SELECT "Project Name" AS project, Address AS address,
                   strftime(sale_date, '%Y-%m-%d') AS sale_date,
                   area_sqft, psf AS unit_psf, price AS transacted_price,
                   tenure_type AS tenure, type_of_sale AS sale_type
            FROM tdf
            WHERE postal_district = {sel_dist}
            ORDER BY sale_date DESC
        """).df()

        st.markdown(f"**Transactions in {DISTRICT_LABELS.get(sel_dist, sel_dist)}**  ·  {len(detail):,} records")
        st.dataframe(detail.style.format({"area_sqft": "{:,.0f}", "unit_psf": "{:,.0f}",
                                          "transacted_price": "{:,.0f}"}),
                     width="stretch", hide_index=True)
        st.download_button("Download transactions (CSV)", detail.to_csv(index=False),
                           file_name=f"transactions_D{sel_dist:02d}.csv", mime="text/csv",
                           key="dl_district_txn")

# ---------------------------------------------------------------- planning area ranking
with t2:
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

    st.markdown("**Avg $PSF by Planning Area** (areas with ≥10 transactions)  ·  click a bar to "
               "see that area's transactions")
    st.caption("Note: areas with fewer than 10 transactions are excluded here because their "
               "averages are statistically unreliable.")
    pick_pa = alt.selection_point(fields=["sub_market"], on="click", empty=False, name="pick_pa")
    pa_chart = alt.Chart(pa).mark_bar().encode(
        y=alt.Y("sub_market:N", sort="-x", title=None),
        x=alt.X("avg_psf:Q", title="Avg $PSF"),
        color=alt.Color("avg_psf:Q", scale=alt.Scale(scheme="viridis"), legend=None),
        opacity=alt.condition(pick_pa, alt.value(1.0), alt.value(0.45)),
        tooltip=["sub_market:N", alt.Tooltip("avg_psf:Q", format=",.0f"),
                 alt.Tooltip("transactions:Q", format=",.0f")]
    ).add_params(pick_pa).properties(height=460)
    pa_event = st.altair_chart(pa_chart, on_select="rerun", key="pa_rank", width="stretch")
    st.dataframe(pa[["sub_market", "avg_psf", "median_psf", "transactions"]]
                .style.format({"avg_psf": "{:,.0f}", "median_psf": "{:,.0f}", "transactions": "{:,.0f}"}),
                width="stretch", hide_index=True)

    # which planning area is selected? (clicked bar, else the top-ranked area)
    sel_pa = None
    try:
        rows = pa_event.selection.get("pick_pa") if pa_event and pa_event.selection else None
        if rows:
            sel_pa = rows[0]["sub_market"]
    except Exception:
        sel_pa = None
    if sel_pa is None:
        sel_pa = pa.iloc[0]["sub_market"]

    pa_tdf = txf[["Project Name", "Address", "sale_date", "area_sqft", "psf", "price",
                 "tenure_type", "type_of_sale", "sub_market"]].copy()
    pa_detail = duckdb.sql("""
        SELECT "Project Name" AS project, Address AS address,
               strftime(sale_date, '%Y-%m-%d') AS sale_date,
               area_sqft, psf AS unit_psf, price AS transacted_price,
               tenure_type AS tenure, type_of_sale AS sale_type
        FROM pa_tdf
        WHERE sub_market = ?
        ORDER BY sale_date DESC
    """, params=[sel_pa]).df()

    st.markdown(f"**Transactions in {sel_pa}**  ·  {len(pa_detail):,} records")
    st.dataframe(pa_detail.style.format({"area_sqft": "{:,.0f}", "unit_psf": "{:,.0f}",
                                         "transacted_price": "{:,.0f}"}),
                width="stretch", hide_index=True)
    st.download_button("Download transactions (CSV)", pa_detail.to_csv(index=False),
                       file_name=f"transactions_{sel_pa.replace(' ', '_')}.csv", mime="text/csv",
                       key="dl_pa_txn")

# ---------------------------------------------------------------- CBD & tenure premium
with t3:
    cbd = txf[txf["sub_market"] == "Downtown Core"]["psf"].median()
    rest = txf[txf["sub_market"] != "Downtown Core"]["psf"].median()
    fh = txf[txf["tenure_type"] == "Freehold"]["psf"].median()
    lh = txf[txf["tenure_type"] == "Leasehold"]["psf"].median()
    m = st.columns(4)
    m[0].metric("Downtown Core median $PSF", f"{cbd:,.0f}")
    m[1].metric("Rest of market median $PSF", f"{rest:,.0f}",
                f"{(cbd/rest-1)*100:+.0f}% CBD premium" if rest else None)
    m[2].metric("Freehold median $PSF", f"{fh:,.0f}")
    m[3].metric("Leasehold median $PSF", f"{lh:,.0f}",
                f"{(fh/lh-1)*100:+.0f}% freehold premium" if lh else None)

    comp = pd.DataFrame({
        "group": ["Downtown Core", "Rest of market", "Freehold", "Leasehold"],
        "median_psf": [cbd, rest, fh, lh],
        "kind": ["Location", "Location", "Tenure", "Tenure"]})
    st.altair_chart(alt.Chart(comp).mark_bar().encode(
        x=alt.X("group:N", title=None, sort=None),
        y=alt.Y("median_psf:Q", title="Median $PSF"),
        color=alt.Color("kind:N", scale=alt.Scale(range=[BLUE, RED]), title=None),
        tooltip=["group:N", alt.Tooltip("median_psf:Q", format=",.0f")]
    ).properties(height=340), width="stretch")

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
        by_band = (mrt_txf.groupby("mrt_band", observed=True)
                  .agg(median_psf=("psf", "median"), transactions=("psf", "size")).reset_index())
        corr = mrt_txf[["dist_to_mrt_km", "psf"]].corr().iloc[0, 1]
        st.caption(f"{len(mrt_txf):,} geocoded transactions, straight-line distance to the "
                   f"nearest MRT/LRT exit. Correlation (distance vs $PSF): **{corr:+.2f}** "
                   f"(negative = pricier closer to a station, as expected).")
        st.altair_chart(alt.Chart(by_band).mark_bar(color=BLUE).encode(
            x=alt.X("mrt_band:N", title="Distance to nearest MRT/LRT", sort=band_labels),
            y=alt.Y("median_psf:Q", title="Median $PSF", scale=alt.Scale(zero=False)),
            tooltip=["mrt_band:N", alt.Tooltip("median_psf:Q", format=",.0f"),
                     alt.Tooltip("transactions:Q", format=",.0f")]
        ).properties(height=320), width="stretch")
