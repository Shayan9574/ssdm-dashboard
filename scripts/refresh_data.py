import os
import json
import shutil
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

# Locate master file across environments. The master workbook is never committed:
# when absent it is bootstrapped from the curated seed, then the six surveillance
# sheets are rebuilt from the CDC Socrata APIs. SSDM_DATA_DIR overrides the data
# folder so the master can live on mounted Google Drive.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SSDM_DATA_DIR", ROOT_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = DATA_DIR / "Research Data.xlsx"
STATUS_FILE = DATA_DIR / "pipeline_status.json"
SEED_FILE = ROOT_DIR / "data" / "curated_seed.xlsx"

if not DATA_FILE.exists():
    legacy = ROOT_DIR / "Research Data.xlsx"
    if legacy.exists():
        DATA_FILE = legacy
        STATUS_FILE = ROOT_DIR / "pipeline_status.json"
    elif SEED_FILE.exists():
        shutil.copy(SEED_FILE, DATA_FILE)
        print(f"Bootstrapped master workbook from curated seed at: {DATA_FILE}")

# Full historical configuration with optimized scope
CDC_ENDPOINTS = {
    "Weekly Hospital Respiratory Dat": {
        "id": "ua7e-t2fy",
        "name": "Weekly Hospital Respiratory Data",
        "network": "CDC NHSN",
        "metrics": "Hospital admissions per 100k, ICU occupancy",
        "extra_filter": None,
        "order_by": "weekendingdate asc"
    },
    "NSSP Emergency Department Visit": {
        "id": "rdmq-nq56",
        "name": "Emergency Department Visits",
        "network": "CDC NSSP",
        "metrics": "ED syndromic visit percentage (%)",
        "extra_filter": "county = 'All'",
        "order_by": "week_end asc"
    },
    "Percent of Tests Positive for V": {
        "id": "seuz-s2cv",
        "name": "Viral Test Positivity",
        "network": "CDC NREVSS",
        "metrics": "PCR laboratory test positivity (%)",
        "extra_filter": None,
        "order_by": "week_end asc"
    },
    "CDC Wastewater Data for SARS-Co": {
        "id": "j9g8-acpt",
        "name": "SARS-CoV-2 Wastewater Load",
        "network": "CDC NWSS",
        "metrics": "Sewershed viral concentration (copies/L)",
        "extra_filter": "lower(state_territory) in ('michigan', 'mi') or state_territory in ('michigan', 'mi', 'Michigan', 'MI')",
        "order_by": "sample_collect_date asc"
    },
    "CDC Epidemic Trends and Rt": {
        "id": "5dqz-y4ea",
        "name": "Epidemic Trends & Rt Velocity",
        "network": "CDC CFA",
        "metrics": "Effective reproduction number (Rt), Growth prob.",
        "extra_filter": "state in ('United States', 'Michigan', 'US', 'National')",
        "order_by": "date asc"
    },
    "NNDSS Weekly Data": {
        "id": "x9gk-5huc",
        "name": "Weekly Notifiable Diseases (NNDSS)",
        "network": "CDC NNDSS",
        "metrics": "Confirmed/probable Meningococcal cases",
        "extra_filter": "label like '%Meningococcal disease%'",
        "order_by": "year asc, week asc"
    }
}

def fetch_full_history_paginated(
    dataset_id: str,
    extra_filter: str = None,
    order_by: str = None,
    page_size: int = 50000,
    max_safety_limit: int = 500000
) -> pd.DataFrame:
    """Paginates through full CDC historical records in 50k chunks."""
    base_url = f"https://data.cdc.gov/resource/{dataset_id}.json"
    all_rows = []
    offset = 0
    headers = {"User-Agent": "SSDM-Public-Health-Dashboard/1.0"}

    print(f"Fetching history from {dataset_id} (Filter: {extra_filter})...")

    while offset < max_safety_limit:
        params = {"$limit": page_size, "$offset": offset}
        if extra_filter:
            params["$where"] = extra_filter
        if order_by:
            params["$order"] = order_by

        response = requests.get(base_url, params=params, headers=headers, timeout=90)
        response.raise_for_status()
        
        batch = response.json()
        if not batch:
            break
            
        all_rows.extend(batch)
        offset += len(batch)
        print(f"  Downloaded {len(all_rows):,} records...")

        if len(batch) < page_size:
            break

    df = pd.DataFrame(all_rows)
    if not df.empty and "__id" not in df.columns:
        df.insert(0, "__id", [f"row-{i}" for i in range(len(df))])

    return df

def update_excel_sheets():
    """Refreshes all surveillance tabs in Research Data.xlsx and writes pipeline_status.json."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Could not locate master file at: {DATA_FILE}")

    print(f"Updating: {DATA_FILE}")
    feed_reports = {}
    total_records = 0
    failures = 0

    with pd.ExcelWriter(DATA_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        for sheet_name, cfg in CDC_ENDPOINTS.items():
            feed_name = cfg["name"]
            try:
                df = fetch_full_history_paginated(
                    dataset_id=cfg["id"],
                    extra_filter=cfg["extra_filter"],
                    order_by=cfg["order_by"]
                )
                
                if df.empty:
                    print(f"⚠️ Warning: No records returned for '{sheet_name}'")
                    feed_reports[feed_name] = {
                        "dataset_id": cfg["id"],
                        "network": cfg["network"],
                        "status": "Warning",
                        "records": 0,
                        "metrics": cfg["metrics"],
                        "error": "No records returned from Socrata endpoint"
                    }
                    continue

                df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
                recs = len(df)
                total_records += recs
                feed_reports[feed_name] = {
                    "dataset_id": cfg["id"],
                    "network": cfg["network"],
                    "status": "Healthy",
                    "records": recs,
                    "metrics": cfg["metrics"],
                    "error": None
                }
                print(f" Successfully updated '{sheet_name}' ({recs:,} records)")
            except Exception as e:
                failures += 1
                print(f"⚠️ Failed to update '{sheet_name}': {e}")
                feed_reports[feed_name] = {
                    "dataset_id": cfg["id"],
                    "network": cfg["network"],
                    "status": "Failed",
                    "records": 0,
                    "metrics": cfg["metrics"],
                    "error": str(e)
                }

    # Save Pipeline Status JSON
    overall = "Healthy" if failures == 0 else ("Partial" if failures < len(CDC_ENDPOINTS) else "Failed")
    status_payload = {
        "last_sync_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "overall_status": overall,
        "total_records": total_records,
        "feeds": feed_reports
    }
    
    with open(STATUS_FILE, "w") as f:
        json.dump(status_payload, f, indent=2)

    print(f"\n Full historical CDC refresh complete. Status saved to: {STATUS_FILE}")

if __name__ == "__main__":
    update_excel_sheets()