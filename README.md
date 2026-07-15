# Singapore Office Price — Data Pipeline & Trends Dashboard

This project turns raw Singapore office-market CSVs into an interactive **price-trends
dashboard** ([app.py](app.py)) and a **data-summary utility** ([data_summary.py](data_summary.py)).
It focuses on **strata office sale transactions** and enriches them with market and macro
context. This README explains exactly how the raw files are cleaned, engineered, and merged to
produce the current output.

> **Scope note.** ~92% of transactions are in the **Central Region**, so all market/macro series
> use the **Central** geography. Whole-building/en-bloc deals are excluded (see cleaning below).

---

## 1. Project layout

```text
PGIM_Office/
  Data/                     # raw source CSVs (read-only; never modified)
  app.py                    # Streamlit trends dashboard  (the main output)
  data_summary.py           # min/median/quartile/outlier summary utility
  outputs/                  # generated summaries (summary_numeric.csv, summary_categorical.csv)
  README.md
```

Run the dashboard:

```bash
streamlit run app.py
```

Run the summary:

```bash
python3 data_summary.py
```

---

## 2. Source files and what we take from each

Everything is joined on a common **quarter** (or **year**) key.

| Raw file (`Data/`) | Column used | Becomes | Native freq |
|---|---|---|---|
| `CommercialTransaction_byProject.csv` | many | transaction records (target + property features) | per-transaction |
| `Property Price Index of Office Space.csv` | *…Central Region (INDEX)* | `price_index` | quarterly |
| `Rental Index of Private Sector Office Space.csv` | *…Central Region (INDEX)* | `rent_index` | quarterly |
| `Vacancy Rate of Private Sector Office Space.csv` | *…Central Region (per cent)* | `vacancy_rate` | quarterly |
| `Private Sector Office Space under Construction:Pipeline:Planned Supply.csv` | *Supply…in the Pipeline* | `supply_pipeline` | quarterly |
| `GDP Growth Rate.csv` | *GDP At Current Market Prices* | `gdp_growth` | quarterly (DOS wide) |
| `CPI quarterly.csv` | *All Items* | `cpi` | quarterly (DOS wide) |
| `Domestic Interest Rates (9).csv` | *Compound SORA - 3 month* | `sora_3m` | daily (MAS) |
| `quarterly overall unemployment rate.csv` | *seasonally_adjusted…* | `unemployment` | quarterly |

`CommercialTransaction_byStreet.csv` is byte-identical to `_byProject` and is ignored.

---

## 3. Data cleaning

### 3.1 Transactions
1. Read with `thousands=','`; parse `Sale Date` as `%d %b %Y`.
2. Keep only `Type of Area == 'Strata'` (removes 74 `Land` / en-bloc rows on a different $PSF basis).
3. **Remove whole-building / mislabeled en-bloc deals** — drop rows where any of:
   - `Area (SQFT) > 30,000`, or
   - `Transacted Price ($) > 50,000,000`, or
   - address contains `ENBLOC`, or
   - `Unit Price ($ PSF) > 10,000` (catches the $538M "strata" SHENTON HOUSE error).
   This matters: it drops ~38 rows and moves average transacted price from ~$3.37M to ~$2.73M.

Result: **~6,859** clean strata transactions, 2010-01 → 2026-06.

### 3.2 Market-context series (tidy quarterly CSVs)
Each is read, `Quarter` (`2010Q1`) parsed to a pandas Period, the **Central** column selected and
coerced to numeric, and reshaped to a long `quarter, value` table (`_load_market`).

### 3.3 Macro series (need reshaping)
- **GDP & CPI** are SingStat "DOS wide" exports (~10 metadata rows, then periods as *columns*).
  `_load_dos_wide` finds the `Data Series` header row, reads from there, selects the target series
  row (`GDP At Current Market Prices` / `All Items`), and transposes the `YYYY nQ` columns into a
  long `quarter, value` table.
- **SORA** is a daily MAS export (6 header lines skipped). We parse `SORA Publication Date`, take
  `Compound SORA - 3 month`, and aggregate **daily → quarterly mean**.
- **Unemployment** is filtered to `residential_status == 'overall'`; the month is mapped to a
  quarter and the seasonally-adjusted rate kept.

---

## 4. Feature engineering (per transaction)

Derived in `load_data`:

| Field | Derivation |
|---|---|
| `price` | `Transacted Price ($)` |
| `psf` | `Unit Price ($ PSF)` (target for the PSF views) |
| `area_sqft` | `Area (SQFT)` |
| `quarter` / `year` / `month` / `quarter_of_year` | from `Sale Date` |
| `sub_market` | `Planning Area` (URA planning areas — shown as "Planning Area" in the UI) |
| `tenure_type` | `Freehold` if `Tenure` contains "Freehold"/"999", else `Leasehold` |
| `size_band` | bins of area: `<=500 / 500-1k / 1k-2k / 2k-5k / >5k` sqft |
| `type_of_sale` | `Type of Sale` (Resale / New Sale / Sub Sale) |
| `floor` | regex `#(\d+)-` from `Address` (~97% coverage), median-imputed otherwise |
| `real_psf` | `psf × 100 / cpi` — CPI-deflated $PSF (2024 base), via a quarter join to `cpi` |

---

## 5. How the files are merged (the join logic)

**Step A — build one market table (`market`).**
Start from `price_index`, then **outer-merge on `quarter`** each of `rent_index`, `vacancy_rate`,
`supply_pipeline`, then `gdp_growth`, `cpi`, `sora_3m` (daily→quarterly), and `unemployment`.
Outer joins keep every quarter any series covers; missing cells stay `NaN`.

```
price_index ─(outer on quarter)─ rent_index ─ vacancy_rate ─ supply_pipeline
            ─ gdp_growth ─ cpi ─ sora_3m(qtr mean) ─ unemployment  ->  market
```

**Step B — deflate.** `cpi` is merged into the transaction table on `quarter` to compute `real_psf`.

**Step C — aggregate transactions (`aggregate`).**
Group the (filtered) transactions by `quarter` **or** `year` and compute:

| Metric | How |
|---|---|
| `volume` | count of transactions |
| `total_area` | sum of `area_sqft` |
| `total_value` | sum of `price` |
| `avg_price` | mean of `price` |
| `median_psf`, `mean_psf` | median / mean of `psf` |
| `median_real_psf` | median of `real_psf` |

**Step D — attach market/macro to each period.**
- *Quarter view:* left-join `market` on `quarter`.
- *Year view:* first collapse `market` to a **yearly mean**, then left-join on `year`.

The result is one row per period with transaction metrics **and** market/macro context — this
feeds every chart and the bottom data table.

---

## 6. Output

### Dashboard ([app.py](app.py))
Top **filter bar** (Year range, Planning Area, Type of sale, Floor range, Tenure, Size band) →
**KPI cards** (with period-on-period deltas) → **pill tabs**:

- **Avg Price** — average transacted price over time (no event lines; it's mix-driven).
- **PSF** — median $PSF over time, with **COVID (2020)** and **Rate-Hike Surge (2022)** markers.
- **Transaction Volume** — toggle between **count / total area (sqft) / total value ($)**.
- **Seasonality** — year × month median-$PSF heatmap + median $PSF by month (peaks highlighted).
- **Macro Factors** — sub-tabs: Price vs CPI / Interest Rate / GDP / Rent Index / Office Price
  Index (dual-axis + correlation).

A **"Break down by Planning Area"** toggle splits the top-6 areas into separate lines. The bottom
**data table** mirrors the aggregated series and is downloadable as CSV.

### Summary ([data_summary.py](data_summary.py))
Prints and writes `outputs/summary_numeric.csv` (count, mean, std, min, quartiles, max, IQR-outlier
counts) and `outputs/summary_categorical.csv` (unique, top value, frequency) for every feature.

---

## 7. Known caveats
- **Central-focused:** market/macro series use the Central geography; other regions are sparse.
- **Latest quarter lag:** the most recent quarter (e.g. 2026Q2) may show `NaN` for GDP/CPI because
  those releases lag transactions.
- **Freehold `building_age`/lease:** only leasehold has a parseable lease start; freehold is
  flagged via `tenure_type` rather than given an age.
- **Not yet added (external):** 10Y SGS yield, STI/S-REIT, SGD FX, office-using employment, BCA
  construction cost, policy events, and OneMap geocoding (`lat/lon`, distances) are planned but not
  in `Data/` yet.
