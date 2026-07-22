"""
Shared data pipeline for the SG Office Analysis app.
Cleans transactions + market/macro series and exposes a cached load_data().
Also provides Singapore postal-district centroids/labels for the geographic page.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "Data"

# Approximate centroids (lat, lon) for Singapore postal districts D01–D28.
DISTRICT_CENTROIDS = {
    1: (1.2830, 103.8510), 2: (1.2760, 103.8450), 3: (1.2870, 103.8270),
    4: (1.2660, 103.8220), 5: (1.2900, 103.7820), 6: (1.2925, 103.8530),
    7: (1.3010, 103.8600), 8: (1.3110, 103.8560), 9: (1.3000, 103.8380),
    10: (1.3130, 103.8070), 11: (1.3200, 103.8400), 12: (1.3300, 103.8560),
    13: (1.3370, 103.8720), 14: (1.3180, 103.8880), 15: (1.3040, 103.9050),
    16: (1.3240, 103.9430), 17: (1.3560, 103.9800), 18: (1.3540, 103.9540),
    19: (1.3700, 103.8970), 20: (1.3620, 103.8480), 21: (1.3380, 103.7770),
    22: (1.3390, 103.7060), 23: (1.3800, 103.7620), 24: (1.3970, 103.7180),
    25: (1.4360, 103.7860), 26: (1.3900, 103.8200), 27: (1.4300, 103.8350),
    28: (1.4050, 103.8700),
}
DISTRICT_LABELS = {
    1: "D01 · Raffles Place, Marina, Cecil", 2: "D02 · Tanjong Pagar, Anson",
    3: "D03 · Tiong Bahru, Queenstown, Alexandra", 4: "D04 · Telok Blangah, HarbourFront, Sentosa",
    5: "D05 · Buona Vista, Clementi, Pasir Panjang, West Coast", 6: "D06 · City Hall, High Street, Beach Road",
    7: "D07 · Bugis, Rochor, Beach Road", 8: "D08 · Little India, Farrer Park",
    9: "D09 · Orchard, River Valley, Cairnhill", 10: "D10 · Tanglin, Holland, Bukit Timah",
    11: "D11 · Newton, Novena, Thomson", 12: "D12 · Balestier, Toa Payoh, Serangoon",
    13: "D13 · Macpherson, Braddell, Potong Pasir", 14: "D14 · Geylang, Eunos, Paya Lebar",
    15: "D15 · Marine Parade, Katong, Joo Chiat", 16: "D16 · Bedok, Upper East Coast",
    17: "D17 · Changi, Loyang, Flora", 18: "D18 · Tampines, Pasir Ris",
    19: "D19 · Serangoon Gardens, Hougang, Punggol, Sengkang", 20: "D20 · Ang Mo Kio, Bishan, Thomson",
    21: "D21 · Upper Bukit Timah, Clementi Park, Ulu Pandan", 22: "D22 · Jurong, Boon Lay, Tuas",
    23: "D23 · Bukit Batok, Bukit Panjang, Choa Chu Kang, Hillview", 24: "D24 · Lim Chu Kang, Tengah",
    25: "D25 · Woodlands, Kranji", 26: "D26 · Upper Thomson, Mandai, Springleaf",
    27: "D27 · Yishun, Sembawang", 28: "D28 · Seletar, Yio Chu Kang",
}


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
    # "Strata" = individual unit sale; "Land" = whole-building/site sale (its $PSF is priced
    # on LAND area, not unit area — not the same metric). Both kept here; the Transaction
    # Type sidebar filter (type_filter, below) picks which the user sees.
    tx["type_of_area"] = tx["Type of Area"]
    # Drops only the mis-recorded SHENTON HOUSE en-bloc row (whole-building $538M sale
    # posted against one unit's floor area, $psf 113,337) — bulk multi-unit purchases
    # (e.g. Solitaire on Cecil, Samsung Hub) have normal $psf and are kept. Never affects
    # Land rows (their $psf, priced on site area, tops out around $39,000).
    tx = tx[tx["Unit Price ($ PSF)"] < 50_000].copy()

    tx["price"] = tx["Transacted Price ($)"]
    tx["psf"] = tx["Unit Price ($ PSF)"]
    tx["area_sqft"] = tx["Area (SQFT)"]
    tx["quarter"] = tx["sale_date"].dt.to_period("Q")
    tx["year"] = tx["sale_date"].dt.year
    tx["month"] = tx["sale_date"].dt.month
    tx["quarter_of_year"] = tx["sale_date"].dt.quarter
    tx["sub_market"] = tx["Planning Area"]
    tx["postal_district"] = pd.to_numeric(tx["Postal District"], errors="coerce").astype("Int64")
    tx["tenure_type"] = np.where(
        tx["Tenure"].str.contains("Freehold|999", case=False, na=False), "Freehold", "Leasehold")
    tx["size_band"] = pd.cut(tx["Area (SQFT)"], [0, 500, 1000, 2000, 5000, np.inf],
                             labels=["<=500", "500-1k", "1k-2k", "2k-5k", ">5k"])
    tx["type_of_sale"] = tx["Type of Sale"]
    tx["floor"] = tx["Address"].str.extract(r"#(\d+)-")[0].astype(float)
    tx["floor_imputed"] = tx["floor"].isna()
    tx["floor"] = tx["floor"].fillna(tx["floor"].median()).astype(int)
    tx["deal_band"] = pd.cut(tx["price"], [0, 5e6, 10e6, np.inf],
                             labels=["<$5M", "$5-10M", ">$10M"])
    # street name: strip the leading house number(s) — some addresses list several,
    # comma-separated, e.g. "175,177 THOMSON ROAD" — and the "#unit" suffix, e.g.
    # "3 SHENTON WAY #24-01" -> "SHENTON WAY"
    tx["street"] = (tx["Address"].str.replace(
        r"^\s*\d+[A-Za-z]?(?:\s*,\s*\d+[A-Za-z]?)*\s*(?:\(ENBLOC\)\s*|ENBLOC\s*)?", "", regex=True)
                    .str.split("#").str[0].str.strip())

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
    # Real GDP growth (chained 2015 dollars), not "At Current Market Prices" (nominal):
    # nominal growth mixes inflation into the signal (e.g. 2021Q2 shows +33.7% nominal).
    for path, series, name in [("GDP Growth Rate.csv", "GDP In Chained (2015) Dollars", "gdp_growth"),
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


def type_filter(tx):
    """Sidebar 'Transaction Type' selector shared by every page (persists across page
    switches via the shared session-state key). Strata = individual unit sales, the norm
    for $PSF analysis. Land = whole-building/site sales, priced per sqft of LAND, not unit
    area — a different metric. The two are never pooled, since their $PSF isn't comparable."""
    choice = st.sidebar.radio(
        "Transaction Type", ["Strata", "Land"], index=0, key="txn_type",
        help="Strata = individual unit sales (used in most charts on this app). "
             "Land = whole-building or site sales — priced per sqft of LAND, not unit area, "
             "so not directly comparable to Strata $PSF.")
    tx = tx[tx["type_of_area"] == choice].copy()
    if choice != "Strata":
        st.sidebar.caption("Land $PSF is priced on site/land area, not unit area — "
                           "not directly comparable to Strata $PSF.")
    return tx, choice
