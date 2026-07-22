"""
One-time/occasional script: geocode every distinct BUILDING (not every transaction —
transactions in the same building share one point, §9 of README) via OneMap's Search
API, caching the result to Data/geocoded_buildings.csv. The dashboard reads that CSV
(backend/data_pipeline.py); it never calls OneMap itself, so Streamlit Cloud needs no
OneMap credentials and the app stays fast/offline-friendly.

Each building gets a `precision` tag (see _geocode_building's docstring for the
3-tier fallback that produces it): "building" (exact block+street or project-name
match), "street" (only the street resolved — the specific building is no longer in
OneMap's address index, almost always because it was sold en-bloc and demolished/
redeveloped since), or "missing" (not even the street resolved).

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


def _parse_address(addr):
    """'37,39 ETC ROBINSON ROAD' -> ('37', 'ROBINSON ROAD'); '10,240(ENBLOC) HOE CHIANG
    ROAD/TANJONG PAGAR ROAD' -> ('10', 'HOE CHIANG ROAD'): keep the first block number
    (a multi-number listing is one building — same geocode point either way, this
    precision doesn't distinguish blocks in one development) and the first street name
    when several are slash-joined (corner buildings), dropping the "(ENBLOC)"/"ETC"
    filler OneMap's search doesn't need. Returns (block_number_or_None, street)."""
    addr = re.sub(r"\(ENBLOC\)|ENBLOC", "", addr, flags=re.I).strip()
    m = re.match(r"(\d+[A-Za-z]?)(?:\s*,\s*\d+[A-Za-z]?)*\s*(?:ETC)?\s*(.*)", addr, flags=re.I)
    if not m:
        return None, addr
    return m.group(1), m.group(2).strip().split("/")[0].strip()


def _geocode_building(project, block_address, token):
    """3-tier fallback, in order of decreasing precision. Every tier is tried and the
    first hit wins — a later tier is strictly less precise, so it's only used when
    every earlier one has genuinely failed (checked directly against OneMap for the
    original 10 misses, §8: not a query-formatting gap — the exact street-level query
    the 2nd tier here uses returned zero results too via block+street/project name).
      1. "block# + street" (e.g. "20 MAXWELL ROAD") -> precision "building"
      2. project name alone (handles addresses OneMap indexes by building name, not
         block number) -> precision "building"
      3. street name alone, no block number -> precision "street" (approximate: places
         the point somewhere on the street, not at the specific building — used only
         for buildings OneMap's current address index has no record of at all, almost
         always because they were sold en-bloc and demolished/redeveloped since)."""
    num, street = _parse_address(block_address)
    query = f"{num} {street}".strip() if num else street
    lat, lon, matched = _search(query, token)
    if lat is not None:
        return lat, lon, matched, query, "building"
    lat, lon, matched = _search(project, token)
    if lat is not None:
        return lat, lon, matched, project, "building"
    if street:
        lat, lon, matched = _search(street, token)
        if lat is not None:
            return lat, lon, matched, street, "street"
    return None, None, None, query, "missing"


def main():
    tx = pd.read_csv(DATA / "CommercialTransaction_byProject.csv", thousands=",")
    block_addr = tx["Address"].str.split("#").str[0].str.strip()
    uniq = (tx.assign(block_address=block_addr)[["Project Name", "block_address"]]
            .drop_duplicates().reset_index(drop=True))
    print(f"{len(uniq)} unique buildings to geocode")

    token = _onemap_token()
    rows, counts = [], {"building": 0, "street": 0, "missing": 0}
    for i, r in uniq.iterrows():
        lat, lon, matched, query, precision = _geocode_building(
            r["Project Name"], r["block_address"], token)
        counts[precision] += 1
        rows.append({"Project Name": r["Project Name"], "block_address": r["block_address"],
                     "query_used": query, "lat": lat, "lon": lon,
                     "matched_address": matched, "precision": precision})
        print(f"[{i + 1}/{len(uniq)}] {precision:8s}  {r['block_address']}")
        time.sleep(0.2)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    total = len(uniq)
    print(f"\nDone: {counts['building']} building-level, {counts['street']} street-level "
         f"(approximate), {counts['missing']} missing entirely — of {total} buildings. "
         f"Saved to {OUT}")


if __name__ == "__main__":
    main()
