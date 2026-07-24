"""
Data summary for the Singapore office-price features available NOW
(transaction-derived + in-folder market/macro series; no external data yet).

Outputs:
  - printed numeric + categorical summaries
  - outputs/summary_numeric.csv, outputs/summary_categorical.csv
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = ROOT / "Data"
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 200, "display.max_columns", 40)


# ---------------------------------------------------------------- loaders
def load_market(path, col, name):
    """Quarterly tidy CSV -> [quarter, name]."""
    df = pd.read_csv(DATA / path)
    df["quarter"] = pd.PeriodIndex(df["Quarter"], freq="Q")
    df[name] = pd.to_numeric(df[col], errors="coerce")
    return df[["quarter", name]]


def load_dos_wide(path, series, name):
    """SingStat DOS wide format (periods as columns) -> [quarter, name]."""
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
    return pd.DataFrame(rec, columns=["quarter", name]).dropna().sort_values("quarter")


# ---------------------------------------------------------------- transactions
tx = pd.read_csv(DATA / "CommercialTransaction_byProject.csv", thousands=",",
                 dtype={"Postal District": str, "Postal Sector": str, "Postal Code": str})
tx["sale_date"] = pd.to_datetime(tx["Sale Date"], format="%d %b %Y")
n0 = len(tx)
tx = tx[tx["Type of Area"] == "Strata"].copy()
n1 = len(tx)
tx = tx[tx["Unit Price ($ PSF)"] <= 10000].copy()          # drop mislabeled en-bloc error
n2 = len(tx)

tx["psf"] = tx["Unit Price ($ PSF)"]
tx["log_psf"] = np.log(tx["psf"])
tx["area_sqft"] = tx["Area (SQFT)"]
tx["log_area"] = np.log(tx["area_sqft"])
tx["size_band"] = pd.cut(tx["area_sqft"], [0, 500, 1000, 2000, 5000, np.inf],
                         labels=["<=500", "500-1k", "1k-2k", "2k-5k", ">5k"])
# Basement units "#B1-.." -> negative floor; multi-floor units (comma- or slash-listed)
# averaged; whole-building rows with no unit -> NaN (not median-filled). Matches the
# pipeline parser in data_pipeline.py.
def _mean_floor(addr):
    floors = []
    for part in re.findall(r"#([^-]+)-", str(addr)):
        nums = [int(n) for n in re.findall(r"\d+", part)]
        if "B" in part.upper():
            floors.extend(-n for n in nums)
        else:
            floors.extend(nums)
    return round(np.mean(floors)) if floors else np.nan
tx["floor"] = tx["Address"].apply(_mean_floor)
tx["high_floor"] = (tx["floor"] >= 20).fillna(False).astype(int)  # NaN floor -> not high
tx["type_of_sale"] = tx["Type of Sale"]

tx["is_freehold"] = tx["Tenure"].str.contains("Freehold|999", case=False, na=False).astype(int)
ls = pd.to_datetime(tx["Tenure"].str.extract(r"from (\d{2}/\d{2}/\d{4})")[0],
                    format="%d/%m/%Y", errors="coerce")
term = tx["Tenure"].str.extract(r"(\d+)\s*yrs")[0].astype(float)
age_years = (tx["sale_date"] - ls).dt.days / 365.25            # leasehold only (NaN for freehold)
tx["building_age"] = age_years
tx["remaining_lease_years"] = term - age_years                 # arithmetic, avoids date overflow
tx.loc[tx["is_freehold"] == 1, "remaining_lease_years"] = 999

tx["street"] = (tx["Address"].str.replace(r"#\S+", "", regex=True)
                .str.replace(r"\bETC\b", "", regex=True, case=False)
                .str.replace(r"^\s*\d+[A-Z]?\s+", "", regex=True)
                .str.replace(r"\s+", " ", regex=True).str.strip())
tx["planning_region"] = tx["Planning Region"]
tx["planning_area"] = tx["Planning Area"]
tx["postal_district"] = tx["Postal District"]
tx["postal_sector"] = tx["Postal Sector"]
tx["is_prime_cbd"] = (tx["Planning Area"] == "Downtown Core").astype(int)

tx["year"] = tx["sale_date"].dt.year
tx["quarter"] = tx["sale_date"].dt.to_period("Q")
tx["month"] = tx["sale_date"].dt.month
tx["quarter_of_year"] = tx["sale_date"].dt.quarter
tx["time_index"] = (tx["sale_date"].dt.year - 2010) * 12 + tx["sale_date"].dt.month - 1
tx["is_covid"] = tx["sale_date"].between("2020-01-01", "2021-12-31").astype(int)
tx["is_rate_hike_cycle"] = tx["sale_date"].between("2022-04-01", "2023-09-30").astype(int)

# ---------------------------------------------------------------- market + macro joins
mkt = [
    load_market("Property Price Index of Office Space.csv",
                "Property Price Index of Office Space in Central Region (INDEX)", "price_index"),
    load_market("Rental Index of Private Sector Office Space.csv",
                "Rental Index of Private Sector Office Space in Central Region (INDEX)", "rent_index"),
    load_market("Vacancy Rate of Private Sector Office Space.csv",
                "Vacancy Rate of Private Sector Office Space in Central Region (per cent)", "vacancy_rate"),
    load_market("VacantSpace.csv",
                "Vacant Private Sector Office Space (Whole Island) ('000 SQ M NETT)", "vacant_stock"),
    load_market("Private Sector Office Space under Construction:Pipeline:Planned Supply.csv",
                "Supply of Private Sector Office Space in the Pipeline ('000 SQ M GROSS)", "supply_pipeline"),
    load_dos_wide("GDP Growth Rate.csv", "GDP At Current Market Prices", "gdp_growth"),
    load_dos_wide("CPI quarterly.csv", "All Items", "cpi"),
]
# SORA (daily -> quarterly mean)
sora = pd.read_csv(DATA / "Domestic Interest Rates (9).csv", skiprows=6)
sora.columns = [str(c).strip() for c in sora.columns]
sora["date"] = pd.to_datetime(sora["SORA Publication Date"], format="%d %b %Y", errors="coerce")
sora["sora_3m"] = pd.to_numeric(sora["Compound SORA - 3 month"], errors="coerce")
sora = sora.dropna(subset=["date", "sora_3m"])
mkt.append(sora.groupby(sora["date"].dt.to_period("Q"))["sora_3m"].mean()
           .reset_index().rename(columns={"date": "quarter"}))
# Unemployment (quarterly)
un = pd.read_csv(DATA / "quarterly overall unemployment rate.csv")
un = un[un["residential_status"] == "overall"].copy()
un["quarter"] = pd.PeriodIndex(pd.to_datetime(un["month"]), freq="Q")
un["unemployment"] = pd.to_numeric(un["seasonally_adjusted_unemployment_rate"], errors="coerce")
mkt.append(un[["quarter", "unemployment"]])

for m in mkt:
    tx = tx.merge(m, on="quarter", how="left")

# ---------------------------------------------------------------- summaries
NUM = ["psf", "log_psf", "area_sqft", "log_area", "floor", "building_age", "time_index",
       "price_index", "rent_index", "vacancy_rate", "vacant_stock", "supply_pipeline",
       "gdp_growth", "cpi", "sora_3m", "unemployment"]
CAT = ["size_band", "high_floor", "type_of_sale", "is_freehold", "is_prime_cbd",
       "planning_region", "planning_area", "postal_district", "quarter_of_year",
       "is_covid", "is_rate_hike_cycle", "year"]


def num_summary(df, cols):
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out = int(((s < lo) | (s > hi)).sum())
        rows.append(dict(feature=c, n=s.count(), missing=int(df[c].isna().sum()),
                         mean=s.mean(), std=s.std(), min=s.min(), p25=q1, median=s.median(),
                         p75=q3, max=s.max(), outliers=out,
                         outlier_pct=round(100 * out / max(len(s), 1), 1)))
    return pd.DataFrame(rows).round(2)


def cat_summary(df, cols):
    rows = []
    for c in cols:
        s = df[c]
        vc = s.value_counts(dropna=False)
        rows.append(dict(feature=c, unique=int(s.nunique()), missing=int(s.isna().sum()),
                         top=str(vc.index[0]), top_freq=int(vc.iloc[0]),
                         top_pct=round(100 * vc.iloc[0] / len(s), 1)))
    return pd.DataFrame(rows)


# remaining_lease_years reported leasehold-only (freehold sentinel 999 excluded)
rly = tx.loc[tx["is_freehold"] == 0, "remaining_lease_years"]

num = num_summary(tx, NUM)
lease_row = num_summary(tx[tx["is_freehold"] == 0], ["remaining_lease_years"])
num = pd.concat([num, lease_row], ignore_index=True)
cat = cat_summary(tx, CAT)

print("=" * 90)
print(f"Transactions: {n0} raw -> {n1} strata -> {n2} after dropping en-bloc error row")
print(f"Date range: {tx['sale_date'].min().date()} to {tx['sale_date'].max().date()}")
print(f"Freehold share: {tx['is_freehold'].mean():.1%} | building_age & remaining_lease are leasehold-only")
print("=" * 90)
print("\nNUMERIC FEATURES (min/quartiles/max + IQR outliers)\n")
print(num.to_string(index=False))
print("\nCATEGORICAL FEATURES\n")
print(cat.to_string(index=False))

num.to_csv(OUT / "summary_numeric.csv", index=False)
cat.to_csv(OUT / "summary_categorical.csv", index=False)
print(f"\nSaved -> {OUT/'summary_numeric.csv'} , {OUT/'summary_categorical.csv'}")
