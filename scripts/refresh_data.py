"""
refresh_data.py

Weekly refresh of the seven CDC surveillance feeds (Socrata API, full
history, paginated). Each feed is written to <data dir>/feeds/<sheet>.parquet.
A feed that fails, returns nothing, or returns fewer than half the records
of the previous good download keeps its previous file, and the failure is
recorded in pipeline_status.json, so the dashboard always runs on the last
good week.

Data dir: $SSDM_DATA_DIR, else <repo>/data.
Optional: $SOCRATA_APP_TOKEN raises the CDC rate limit.
Optional: --workbook also writes the feeds into the legacy "Research
Data.xlsx" (Colab compatibility).

Usage:  python scripts/refresh_data.py [--workbook]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SSDM_DATA_DIR", ROOT / "data"))
FEEDS_DIR = DATA_DIR / "feeds"
STATUS_FILE = DATA_DIR / "pipeline_status.json"

CDC_ENDPOINTS = {
    "Weekly Hospital Respiratory Dat": {
        "id": "ua7e-t2fy", "name": "Weekly Hospital Respiratory Data", "network": "CDC NHSN",
        "metrics": "Hospital admissions per 100k, ICU share, occupancy",
        "where": None, "order": "weekendingdate asc"},
    "NSSP Emergency Department Visit": {
        "id": "rdmq-nq56", "name": "Emergency Department Visits", "network": "CDC NSSP",
        "metrics": "ED syndromic visit share (%)", "where": "county = 'All'", "order": "week_end asc"},
    "Percent of Tests Positive for V": {
        "id": "seuz-s2cv", "name": "Viral Test Positivity", "network": "CDC NREVSS",
        "metrics": "PCR test positivity (%), national", "where": None, "order": "week_end asc"},
    "CDC Wastewater Data for SARS-Co": {
        "id": "j9g8-acpt", "name": "SARS-CoV-2 Wastewater", "network": "CDC NWSS",
        "metrics": "Sewershed concentration (copies/L), Michigan sewersheds",
        "where": "lower(state_territory) in ('michigan', 'mi')", "order": "sample_collect_date asc"},
    "CDC Epidemic Trends and Rt": {
        "id": "5dqz-y4ea", "name": "Epidemic Trends and Rt", "network": "CDC CFA",
        "metrics": "Effective reproduction number, growth probability",
        "where": "state in ('United States', 'Michigan', 'US', 'National')", "order": "date asc"},
    "RESP-NET Rates and Clinical Dat": {
        "id": "kvib-3txy", "name": "RESP-NET Hospitalization Rates", "network": "CDC RESP-NET",
        "metrics": "Weekly hospitalization rates per 100k by age",
        "where": "state = 'Overall' and sex = 'All' and race = 'All'", "order": "date asc"},
    "NNDSS Weekly Data": {
        "id": "x9gk-5huc", "name": "Weekly Notifiable Diseases (NNDSS)", "network": "CDC NNDSS",
        "metrics": "Meningococcal cases", "where": "label like '%Meningococcal disease%'",
        "order": "year asc, week asc"},
}


def fetch(dataset_id: str, where=None, order=None, page=50000, cap=2_000_000) -> pd.DataFrame:
    url = f"https://data.cdc.gov/resource/{dataset_id}.json"
    headers = {"User-Agent": "SSDM-Public-Health-Dashboard/2.0"}
    if os.environ.get("SOCRATA_APP_TOKEN"):
        headers["X-App-Token"] = os.environ["SOCRATA_APP_TOKEN"]
    rows, offset = [], 0
    while offset < cap:
        params = {"$limit": page, "$offset": offset}
        if where:
            params["$where"] = where
        if order:
            params["$order"] = order
        for attempt in range(4):
            try:
                r = requests.get(url, params=params, headers=headers, timeout=120)
                r.raise_for_status()
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(10 * (attempt + 1))
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        print(f"  {dataset_id}: {len(rows):,} records", flush=True)
        if len(batch) < page:
            break
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str)
    return df


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", action="store_true")
    args = ap.parse_args()
    FEEDS_DIR.mkdir(parents=True, exist_ok=True)
    prev = json.loads(STATUS_FILE.read_text()) if STATUS_FILE.exists() else {}
    report, failures = {}, 0
    for sheet, cfg in CDC_ENDPOINTS.items():
        out = FEEDS_DIR / f"{sheet}.parquet"
        old_n = prev.get("feeds", {}).get(cfg["name"], {}).get("records", 0) or 0
        try:
            df = fetch(cfg["id"], cfg["where"], cfg["order"])
            if df.empty:
                raise RuntimeError("no records returned")
            if old_n and len(df) < 0.5 * old_n:
                raise RuntimeError(f"{len(df)} records, fewer than half of the previous {old_n}")
            df.to_parquet(out, index=False)
            report[cfg["name"]] = {"sheet": sheet, "dataset_id": cfg["id"], "network": cfg["network"],
                                   "status": "Healthy", "records": len(df), "metrics": cfg["metrics"],
                                   "error": None}
            print(f"updated {sheet}: {len(df):,} records")
        except Exception as e:  # keep the previous good file
            failures += 1
            report[cfg["name"]] = {"sheet": sheet, "dataset_id": cfg["id"], "network": cfg["network"],
                                   "status": "Failed, previous data kept" if out.exists() else "Failed",
                                   "records": old_n, "metrics": cfg["metrics"], "error": str(e)[:300]}
            print(f"FAILED {sheet}: {e}", file=sys.stderr)
    status = {
        "last_sync_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "overall_status": "Healthy" if failures == 0 else ("Partial" if failures < len(CDC_ENDPOINTS) else "Failed"),
        "total_records": sum(v["records"] or 0 for v in report.values()),
        "feeds": report,
        "files": {p.name: {"sha256": _sha(p), "bytes": p.stat().st_size} for p in sorted(FEEDS_DIR.glob("*.parquet"))},
    }
    STATUS_FILE.write_text(json.dumps(status, indent=2))
    if args.workbook:
        wb = DATA_DIR / "Research Data.xlsx"
        seed = ROOT / "data" / "curated_seed.xlsx"
        if not wb.exists() and seed.exists():
            import shutil
            shutil.copy(seed, wb)
        with pd.ExcelWriter(wb, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
            for sheet in CDC_ENDPOINTS:
                p = FEEDS_DIR / f"{sheet}.parquet"
                if p.exists():
                    pd.read_parquet(p).to_excel(w, sheet_name=sheet, index=False, startrow=1)
    print(f"refresh complete: {status['overall_status']}")
    return 0 if failures < len(CDC_ENDPOINTS) else 1


if __name__ == "__main__":
    sys.exit(main())
