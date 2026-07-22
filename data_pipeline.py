"""
Shared data pipeline for the SG Office Analysis app.
Cleans transactions + market/macro series and exposes a cached load_data().
Also provides Singapore postal-district centroids/labels for the geographic page.
"""
import glob
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


def _load_dos_wide_monthly(path, series, name):
    """Same SingStat Table Builder 'wide' layout as _load_dos_wide, but columns are
    calendar months (e.g. '2026 May') instead of quarters ('2026 1Q')."""
    raw = pd.read_csv(DATA / path, header=None, dtype=str)
    hdr = raw.index[raw[0].astype(str).str.strip() == "Data Series"][0]
    df = pd.read_csv(DATA / path, skiprows=hdr)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "series"})
    row = df[df["series"].astype(str).str.strip() == series]
    rec = []
    for col, val in row.iloc[0, 1:].items():
        m = re.match(r"(\d{4})\s*([A-Za-z]{3})", str(col).strip())
        if m:
            period = pd.to_datetime(f"{m.group(1)}-{m.group(2)}-01",
                                    format="%Y-%b-%d", errors="coerce").to_period("M")
            rec.append((period, pd.to_numeric(val, errors="coerce")))
    return pd.DataFrame(rec, columns=["month", name]).dropna()


def _load_mas_monthly(path, name, header_hint):
    """MAS 'Financial Database' monthly export: a single header line containing
    `header_hint`, then rows of (year — blank-filled after January, month abbrev,
    value). Used for the SGS 10-year bond yield and the SGD/USD exchange rate, both
    single-series exports (verified: exactly 3 columns, no other tenor/currency mixed
    in). Monthly SGS yield is MAS's own average-of-daily-bids for the month (not
    re-derived here); monthly FX is MAS's end-of-period convention — both used as
    published, not recomputed, to avoid silently changing MAS's own methodology."""
    with open(DATA / path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    hdr = next(i for i, l in enumerate(lines) if header_hint.lower() in l.lower())
    raw = pd.read_csv(DATA / path, skiprows=hdr + 1, header=None,
                      names=["yr", "mth", name], dtype=str, on_bad_lines="skip")
    raw["yr"] = raw["yr"].ffill()
    raw = raw[raw["yr"].str.match(r"^\d{4}$", na=False) & raw["mth"].notna()]
    raw["month"] = pd.to_datetime(raw["yr"] + "-" + raw["mth"], format="%Y-%b",
                                  errors="coerce").dt.to_period("M")
    raw[name] = pd.to_numeric(raw[name], errors="coerce")
    return raw.dropna(subset=["month", name])[["month", name]]


def _load_construction_materials():
    """Raw monthly market prices, one column per material — deliberately NOT combined
    into a single 'construction cost index'. The five materials are on different
    scales/units (e.g. steel ~$700 vs granite ~$25) and SingStat does not publish
    weights here, unlike BCA's own (weighted) Tender Price Index — averaging them
    unweighted would fabricate a precision we don't have. Use whichever material(s)
    are relevant to a given analysis explicitly, rather than a blended figure."""
    df = pd.read_csv(DATA / "Construction Material Market Prices Monthly.csv")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "material"})
    long = df.melt(id_vars="material", var_name="col", value_name="value")
    long["month"] = pd.to_datetime(long["col"], format="%Y%b", errors="coerce").dt.to_period("M")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["month", "value"])
    key = {"Cement In Bulk (Ordinary Portland Cement)": "price_cement",
           "Steel Reinforcement Bars (16-32mm High Tensile)": "price_steel_rebar",
           "Granite (20mm Aggregate)": "price_granite",
           "Concreting Sand": "price_sand",
           "Ready Mixed Concrete": "price_concrete"}
    out = long.pivot_table(index="month", columns="material", values="value").reset_index()
    out = out.rename(columns=key)
    keep = ["month"] + [c for c in key.values() if c in out.columns]
    return out[keep]


@st.cache_data
def load_market_monthly():
    """Natively monthly market/macro series — NOT derived by forward-filling quarterly
    data (see market_monthly() for that path). Each source here actually publishes at
    monthly frequency, so every value is a real observation, not a repeated step."""
    m = _load_dos_wide_monthly("CPI monthly.csv", "All Items", "cpi")

    sora = pd.read_csv(DATA / "Domestic Interest Rates (9).csv", skiprows=6)
    sora.columns = [str(c).strip() for c in sora.columns]
    sora["date"] = pd.to_datetime(sora["SORA Publication Date"], format="%d %b %Y", errors="coerce")
    sora["sora_3m"] = pd.to_numeric(sora["Compound SORA - 3 month"], errors="coerce")
    sora = sora.dropna(subset=["date", "sora_3m"])
    sm = sora.groupby(sora["date"].dt.to_period("M"))["sora_3m"].mean().reset_index()
    sm.columns = ["month", "sora_3m"]
    m = m.merge(sm, on="month", how="outer")

    m = m.merge(_load_mas_monthly(
        "SGS - Historical Prices and Yields - Benchmark Issues (Monthly).csv",
        "sgs_10y_yield", "10-Year Bond Yield"), on="month", how="outer")
    m = m.merge(_load_mas_monthly(
        "Exchange Rates (Monthly).csv",
        "sgd_usd_fx", "S$ Per Unit of US Dollar"), on="month", how="outer")
    m = m.merge(_load_construction_materials(), on="month", how="outer")
    return m.sort_values("month").reset_index(drop=True)


@st.cache_data
def load_employment_annual():
    """Annual PMET (Professionals, Managers, Executives & Technicians) resident
    employment level — MOM's standard proxy for office-using employment (excludes
    clerical, service/sales, craft, plant/machine, cleaner/labourer occupations).
    Kept as its own annual-frequency table rather than forward-filled into the
    quarter/month tables here: annual -> quarter is a much coarser assumption than
    quarter -> month, so any expansion should be done explicitly where it's used and
    clearly labelled, not baked silently into the shared pipeline.
    'Annual Employment Level by Industry.xlsx' is intentionally NOT used: it is a raw
    SingStat Table Builder interactive-export artifact (SSIC crosswalk/instruction
    sheets, no single clean table) and is redundant with this cleaner occupation-level
    series for a level-type employment metric."""
    o = pd.read_csv(DATA / "Number of Employed Residents by Occupation.csv")
    pmet_occupations = {"managers & administrators (including working proprietors)",
                        "professionals", "associate professionals & technicians"}
    pmet = (o[o["occupation"].isin(pmet_occupations)]
            .groupby("year")["employed"].sum().reset_index()
            .rename(columns={"employed": "pmet_employment"}))
    return pmet


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

    # Office-using services employment change (quarterly, '000 persons, net change) —
    # Finance & Insurance + Real Estate + Professional Services. SSIC industry labels
    # were revised twice across this 1991-2026 window; verified non-overlapping by
    # quarter (no double-count risk): "financial services" (1991-2008) was renamed
    # "financial and insurance services" (2009-2026); "business and real estate
    # services" (1991-2001) split into "real estate and leasing services" (2002-2008)
    # then "real estate services" (2009-2026), with "professional services" carved out
    # as its own category from 2002 onward (no data before then). Summing every label
    # per quarter is therefore safe.
    emp_path = glob.glob(str(DATA / "Quarterly Employ*ment Change by Industry.csv"))[0]
    emp = pd.read_csv(emp_path)
    office_services = {"financial services", "financial and insurance services",
                       "business and real estate services", "real estate and leasing services",
                       "real estate services", "professional services"}
    emp_office = emp[emp["industry2"].isin(office_services)].copy()
    emp_office["employment_change"] = pd.to_numeric(emp_office["employment_change"], errors="coerce")
    eq = (emp_office.groupby("quarter")["employment_change"].sum().reset_index()
          .rename(columns={"employment_change": "office_employment_chg"}))
    eq["quarter"] = pd.PeriodIndex(eq["quarter"], freq="Q")
    market = market.merge(eq, on="quarter", how="outer")

    tx = tx.merge(market[["quarter", "cpi"]], on="quarter", how="left")
    tx["real_psf"] = tx["psf"] * 100 / tx["cpi"]
    return tx, market.sort_values("quarter")


def market_monthly(market):
    """Expand the quarterly market table to monthly by forward-fill: each quarter's
    value is repeated across its 3 calendar months (a step function), since every
    market/macro series in `market` is quarterly-sourced — there is no genuine monthly
    observation to show. Callers must not present this as a true monthly reading."""
    m = market.dropna(subset=["quarter"]).copy()
    m["month"] = m["quarter"].dt.to_timestamp(how="start").dt.to_period("M")
    frames = [m.assign(month=m["month"] + i) for i in range(3)]
    out = pd.concat(frames, ignore_index=True).drop(columns="quarter")
    return out.sort_values("month").reset_index(drop=True)


def downsample_market_monthly(mm, to):
    """Aggregate the natively-monthly table (load_market_monthly()) DOWN to quarter or
    year via mean — the safe direction (fine -> coarse never needs an assumption,
    unlike market_monthly()'s forward-fill). `to` is 'quarter' or 'year'."""
    m = mm.copy()
    if to == "quarter":
        m["quarter"] = m["month"].dt.asfreq("Q")
    else:
        m["year"] = m["month"].dt.year
    return m.drop(columns="month").groupby(to).mean(numeric_only=True).reset_index()


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
