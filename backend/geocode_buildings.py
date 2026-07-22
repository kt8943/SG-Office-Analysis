"""
One-time/occasional script: geocode every distinct BUILDING (not every transaction —
transactions in the same building share one point, §9 of README) via OneMap's Search
API, caching the result to Data/geocoded_buildings.csv. The dashboard reads that CSV
(backend/data_pipeline.py); it never calls OneMap itself, so Streamlit Cloud needs no
OneMap credentials and the app stays fast/offline-friendly.

Re-run this whenever CommercialTransaction_byProject.csv gains new buildings:
    python3 backend/geocode_buildings.py

Needs ONEMAP_EMAIL / ONEMAP_PASSWORD in .streamlit/secrets.toml. Password auth (not a
pasted token) is used deliberately: OneMap's JWT expires after ~3 days, so a token
would go stale between runs; exchanging email+password for a fresh token every run
doesn't have that problem.
"""
import re
import time
import tomllib
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
DATA = ROOT / "Data"
OUT = DATA / "geocoded_buildings.csv"
AUTH_URL = "https://www.onemap.gov.sg/api/auth/post/getToken"
SEARCH_URL = "https://www.onemap.gov.sg/api/common/elastic/search"


def _onemap_token():
    with open(ROOT / ".streamlit" / "secrets.toml", "rb") as f:
        secrets = tomllib.load(f)
    resp = requests.post(AUTH_URL, json={"email": secrets["ONEMAP_EMAIL"],
                                         "password": secrets["ONEMAP_PASSWORD"]}, timeout=15)
    resp.raise_for_status()
    return resp.json()["access_token"]


def _search(query, token):
    r = requests.get(SEARCH_URL, params={"searchVal": query, "returnGeom": "Y",
                                         "getAddrDetails": "Y", "pageNum": 1},
                     headers={"Authorization": f"Bearer {token}"}, timeout=15)
    r.raise_for_status()
    d = r.json()
    if d.get("found", 0) > 0:
        top = d["results"][0]
        return float(top["LATITUDE"]), float(top["LONGITUDE"]), top["ADDRESS"]
    return None, None, None


def _geocode_query(addr):
    """'37,39 ETC ROBINSON ROAD' / '10,240(ENBLOC) HOE CHIANG ROAD/TANJONG PAGAR ROAD'
    -> '37 ROBINSON ROAD' / '10 HOE CHIANG ROAD': keep the first block number (a
    multi-number listing is one building — same geocode point either way, §-level
    precision doesn't distinguish blocks in one development) and the first street name
    when several are slash-joined (corner buildings), dropping the "(ENBLOC)"/"ETC"
    filler OneMap's search doesn't need."""
    addr = re.sub(r"\(ENBLOC\)|ENBLOC", "", addr, flags=re.I).strip()
    m = re.match(r"(\d+[A-Za-z]?)(?:\s*,\s*\d+[A-Za-z]?)*\s*(?:ETC)?\s*(.*)", addr, flags=re.I)
    if not m:
        return addr
    num, rest = m.group(1), m.group(2).strip().split("/")[0].strip()
    return f"{num} {rest}".strip()


def main():
    tx = pd.read_csv(DATA / "CommercialTransaction_byProject.csv", thousands=",")
    block_addr = tx["Address"].str.split("#").str[0].str.strip()
    uniq = (tx.assign(block_address=block_addr)[["Project Name", "block_address"]]
            .drop_duplicates().reset_index(drop=True))
    print(f"{len(uniq)} unique buildings to geocode")

    token = _onemap_token()
    rows, hits = [], 0
    for i, r in uniq.iterrows():
        query = _geocode_query(r["block_address"])
        lat, lon, matched = _search(query, token)
        if lat is None:
            lat, lon, matched = _search(r["Project Name"], token)
        hits += lat is not None
        rows.append({"Project Name": r["Project Name"], "block_address": r["block_address"],
                     "query_used": query, "lat": lat, "lon": lon, "matched_address": matched})
        print(f"[{i + 1}/{len(uniq)}] {'OK  ' if lat is not None else 'MISS'}  {r['block_address']}")
        time.sleep(0.2)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nDone: {hits}/{len(uniq)} geocoded ({hits / len(uniq) * 100:.1f}%). Saved to {OUT}")


if __name__ == "__main__":
    main()
