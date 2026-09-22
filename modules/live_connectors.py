import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

DATA_FILE_PATHS = [
    Path(__file__).resolve().parent.parent / "data" / "Research Data.xlsx",
    Path(__file__).resolve().parent.parent / "Research Data.xlsx",
    Path("data/Research Data.xlsx"),
    Path("Research Data.xlsx")
]

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia", "PR": "Puerto Rico", "US": "United States", "USA": "United States",
    "NATIONAL": "United States"
}

NAME_TO_CODE = {v.lower(): k for k, v in STATE_NAMES.items()}
NAME_TO_CODE.update({k.lower(): k for k in STATE_NAMES.keys()})

def get_data_filepath() -> Path:
    for p in DATA_FILE_PATHS:
        if p.exists():
            return p
    return DATA_FILE_PATHS[0]

def get_state_code(jur: str) -> str:
    j = str(jur).strip().lower()
    if j in {"national", "us", "usa", "united states", "all"}:
        return "US"
    return NAME_TO_CODE.get(j, jur.upper()[:2])

def get_state_full(jur: str) -> str:
    code = get_state_code(jur)
    if code == "US":
        return "United States"
    return STATE_NAMES.get(code, jur.title())

@st.cache_data(ttl=3600)
def load_raw_sheet(sheet_name: str) -> pd.DataFrame:
    fp = get_data_filepath()
    df = pd.read_excel(fp, sheet_name=sheet_name, header=1)
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df

def extract_completed_series(
    df: pd.DataFrame,
    date_col: str,
    lag_complete_weeks: int = 1
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    if df.empty or date_col not in df.columns:
        return df, None

    df_sorted = df.copy()
    df_sorted[date_col] = pd.to_datetime(df_sorted[date_col], errors="coerce")
    df_sorted = df_sorted.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    if df_sorted.empty:
        return df_sorted, None

    target_idx = max(0, len(df_sorted) - 1 - lag_complete_weeks)
    return df_sorted, df_sorted.iloc[target_idx]

# --- 1. HOSPITAL ADMISSIONS & OCCUPANCY ---
def get_hospital_respiratory_surveillance(jurisdiction: str = "National", lag_complete_weeks: int = 1) -> Dict[str, Any]:
    raw = load_raw_sheet("Weekly Hospital Respiratory Dat")
    code = get_state_code(jurisdiction)

    if code == "US":
        df_jur = raw[raw["jurisdiction"].astype(str).str.upper().isin(["USA", "US", "NATIONAL"])]
    else:
        df_jur = raw[raw["jurisdiction"].astype(str).str.upper() == code]

    ts, latest = extract_completed_series(df_jur, "weekendingdate", lag_complete_weeks)
    if latest is None:
        return {}

    return {
        "covid_adm_per100k": pd.to_numeric(latest.get("totalconfc19newadmper100k"), errors="coerce"),
        "flu_adm_per100k": pd.to_numeric(latest.get("totalconfflunewadmper100k"), errors="coerce"),
        "rsv_adm_per100k": pd.to_numeric(latest.get("totalconfrsvnewadmper100k"), errors="coerce"),
        "covid_icu_rate": pd.to_numeric(latest.get("pctconfc19hosppatsicu"), errors="coerce"),
        "flu_icu_rate": pd.to_numeric(latest.get("pctconffluhosppatsicu"), errors="coerce"),
        "rsv_icu_rate": pd.to_numeric(latest.get("pctconfrsvhosppatsicu"), errors="coerce"),
        "covid_ped_adm": pd.to_numeric(latest.get("numconfc19newadmped0to4per100k"), errors="coerce"),
        "flu_ped_adm": pd.to_numeric(latest.get("numconfflunewadmped0to4per100k"), errors="coerce"),
        "rsv_ped_adm": pd.to_numeric(latest.get("numconfrsvnewadmped0to4per100k"), errors="coerce"),
        "covid_ger_adm": pd.to_numeric(latest.get("numconfc19newadmadult75plusper100k"), errors="coerce"),
        "flu_ger_adm": pd.to_numeric(latest.get("numconfflunewadmadult75plusper100k"), errors="coerce"),
        "rsv_ger_adm": pd.to_numeric(latest.get("numconfrsvnewadmadult75plusper100k"), errors="coerce"),
        "inpatient_occ": pd.to_numeric(latest.get("pctinptbedsocc"), errors="coerce"),
        "icu_occ": pd.to_numeric(latest.get("pcticubedsocc"), errors="coerce"),
        "as_of_date": str(latest.get("weekendingdate"))[:10]
    }

def get_hospital_admission_trends(jurisdiction: str = "National") -> pd.DataFrame:
    """Returns weekly new hospital admissions per 100k time-series for COVID-19, Influenza, and RSV."""
    raw = load_raw_sheet("Weekly Hospital Respiratory Dat")
    code = "USA" if jurisdiction in ["National", "US", "USA"] else jurisdiction.upper()[:2]
    sub = raw[raw["jurisdiction"].astype(str).str.upper() == code].copy()
    sub["Date"] = pd.to_datetime(sub["weekendingdate"], errors="coerce")
    sub = sub.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    df = pd.DataFrame({
        "COVID-19": pd.to_numeric(sub["totalconfc19newadmper100k"], errors="coerce"),
        "Influenza": pd.to_numeric(sub["totalconfflunewadmper100k"], errors="coerce"),
        "RSV": pd.to_numeric(sub["totalconfrsvnewadmper100k"], errors="coerce")
    })
    return df.dropna(how="all")

def get_hospital_occupancy_trends(jurisdiction: str = "National") -> pd.DataFrame:
    """Returns weekly hospital bed and ICU occupancy rate trends."""
    raw = load_raw_sheet("Weekly Hospital Respiratory Dat")
    code = "USA" if jurisdiction in ["National", "US", "USA"] else jurisdiction.upper()[:2]
    sub = raw[raw["jurisdiction"].astype(str).str.upper() == code].copy()
    sub["Date"] = pd.to_datetime(sub["weekendingdate"], errors="coerce")
    sub = sub.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    df = pd.DataFrame({
        "Inpatient Bed Occupancy (%)": pd.to_numeric(sub["pctinptbedsocc"], errors="coerce"),
        "ICU Bed Occupancy (%)": pd.to_numeric(sub["pcticubedsocc"], errors="coerce")
    })
    return df.dropna(how="all")

# --- 2. EMERGENCY DEPARTMENT VISITS ---
def get_ed_visit_surveillance(jurisdiction: str = "National", lag_complete_weeks: int = 1) -> Dict[str, Any]:
    raw = load_raw_sheet("NSSP Emergency Department Visit")
    full_name = get_state_full(jurisdiction)

    if jurisdiction.lower() in {"national", "us", "usa", "united states"}:
        df_jur = raw[(raw["geography"].astype(str).str.lower().isin(["united states", "national", "us"])) & (raw["county"] == "All")]
    else:
        df_jur = raw[(raw["geography"].astype(str).str.lower() == full_name.lower()) & (raw["county"] == "All")]

    ts, latest = extract_completed_series(df_jur, "week_end", lag_complete_weeks)
    if latest is None:
        return {}

    return {
        "covid_ed_pct": pd.to_numeric(latest.get("percent_visits_covid"), errors="coerce"),
        "flu_ed_pct": pd.to_numeric(latest.get("percent_visits_influenza"), errors="coerce"),
        "rsv_ed_pct": pd.to_numeric(latest.get("percent_visits_rsv"), errors="coerce"),
        "as_of_date": str(latest.get("week_end"))[:10]
    }

def get_ed_visit_trends(jurisdiction: str = "National") -> pd.DataFrame:
    """Returns weekly ED syndromic visit percentage trends."""
    raw = load_raw_sheet("NSSP Emergency Department Visit")
    geo = "United States" if jurisdiction in ["National", "US", "USA"] else "Michigan"
    sub = raw[(raw["geography"].astype(str).str.lower() == geo.lower()) & (raw["county"] == "All")].copy()
    sub["Date"] = pd.to_datetime(sub["week_end"], errors="coerce")
    sub = sub.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    df = pd.DataFrame({
        "COVID-19": pd.to_numeric(sub["percent_visits_covid"], errors="coerce"),
        "Influenza": pd.to_numeric(sub["percent_visits_influenza"], errors="coerce"),
        "RSV": pd.to_numeric(sub["percent_visits_rsv"], errors="coerce")
    })
    return df.dropna(how="all")

# --- 3. LABORATORY TEST POSITIVITY ---
def get_test_positivity_trends() -> pd.DataFrame:
    """Returns weekly PCR laboratory test positivity trends across respiratory pathogens."""
    raw = load_raw_sheet("Percent of Tests Positive for V")
    piv = raw.pivot_table(index="week_end", columns="pathogen", values="percent_test_positivity", aggfunc="first")
    piv.index = pd.to_datetime(piv.index)
    piv.index.name = "Date"
    return piv.sort_index().dropna(how="all")

# --- 4. EPIDEMIC TRENDS & Rt ---
def get_rt_surveillance(jurisdiction: str = "National", lag_complete_weeks: int = 1) -> Dict[str, Any]:
    raw = load_raw_sheet("CDC Epidemic Trends and Rt")
    full_name = get_state_full(jurisdiction)
    
    if jurisdiction.lower() in {"national", "us", "usa", "united states"}:
        df_jur = raw[raw["state"].astype(str).str.lower().isin(["united states", "national", "us"])]
    else:
        df_jur = raw[raw["state"].astype(str).str.lower() == full_name.lower()]

    out = {}
    for disease in df_jur["disease"].dropna().unique():
        sub = df_jur[df_jur["disease"] == disease]
        ts, latest = extract_completed_series(sub, "date", lag_complete_weeks)
        if latest is not None:
            out[str(disease).strip()] = {
                "median": pd.to_numeric(latest.get("median"), errors="coerce"),
                "p_growing": pd.to_numeric(latest.get("p_growing"), errors="coerce"),
                "as_of_date": str(latest.get("date"))[:10]
            }
    return out

def get_rt_trends(jurisdiction: str = "National") -> pd.DataFrame:
    """
    Returns daily Rt transmission velocity estimates.
    Deduplicates multiple model run snapshots per date by taking the median estimate.
    """
    raw = load_raw_sheet("CDC Epidemic Trends and Rt")
    geo = "United States" if jurisdiction in ["National", "US", "USA"] else "Michigan"
    sub = raw[raw["state"].astype(str).str.lower() == geo.lower()].copy()
    sub["Date"] = pd.to_datetime(sub["date"], errors="coerce")
    
    piv = sub.groupby(["Date", "disease"])["median"].median().unstack("disease")
    return piv.sort_index().dropna(how="all")

# --- 5. WASTEWATER SURVEILLANCE ---
def get_wastewater_surveillance(jurisdiction: str = "National", lag_complete_weeks: int = 1) -> Dict[str, Any]:
    raw = load_raw_sheet("CDC Wastewater Data for SARS-Co")
    code = get_state_code(jurisdiction)

    if code == "US":
        df_jur = raw
    else:
        # Case-insensitive match for MI / Michigan
        df_jur = raw[raw["state_territory"].astype(str).str.strip().str.lower().isin(["mi", "michigan"])]

    ts, latest = extract_completed_series(df_jur, "sample_collect_date", lag_complete_weeks)
    if latest is None:
        return {}

    return {
        "viral_conc": pd.to_numeric(latest.get("pcr_target_avg_conc_lin"), errors="coerce"),
        "as_of_date": str(latest.get("sample_collect_date"))[:10]
    }

def get_wastewater_trends(jurisdiction: str = "National") -> pd.DataFrame:
    """
    Returns sewershed viral concentration trend (SARS-CoV-2 copies/L).
    Uses a 7-day rolling median across reporting sewersheds.
    """
    raw = load_raw_sheet("CDC Wastewater Data for SARS-Co")
    code = get_state_code(jurisdiction)

    if code == "US":
        sub = raw.copy()
    else:
        # Case-insensitive match for MI / Michigan
        sub = raw[raw["state_territory"].astype(str).str.strip().str.lower().isin(["mi", "michigan"])].copy()
    
    sub["Date"] = pd.to_datetime(sub["sample_collect_date"], errors="coerce")
    daily_median = sub.groupby("Date")["pcr_target_avg_conc_lin"].median()
    smoothed = daily_median.rolling(7, min_periods=1).mean().to_frame(name="SARS-CoV-2 Viral Load (copies/L)")
    return smoothed.sort_index().dropna(how="all")

def get_meningitis_trends(jurisdiction: str = "National") -> pd.DataFrame:
    """
    Returns weekly Meningococcal disease case counts alongside a 4-week moving average
    to smooth low-incidence reporting batching.
    """
    raw = load_raw_sheet("NNDSS Weekly Data")
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()

    # 1. Map Socrata API field names
    cols = {str(c).lower().strip().replace(" ", "_").replace("-", "_"): c for c in df.columns}
    year_col = next((cols[k] for k in cols if k in ["year", "mmwr_year", "mmwryear"]), None)
    week_col = next((cols[k] for k in cols if k in ["week", "mmwr_week", "mmwrweek"]), None)
    area_col = next((cols[k] for k in cols if k in ["states", "reporting_area", "location", "jurisdiction", "state"]), None)
    label_col = next((cols[k] for k in cols if k in ["label", "disease", "condition", "pathogen"]), None)
    case_col = next((cols[k] for k in cols if k in ["m1", "current_week", "cases", "count", "num_cases", "data_value"]), None)

    if not year_col or not week_col or not area_col or not case_col:
        return pd.DataFrame()

    # 2. Prevent sub-serogroup double-counting (isolate 'All serogroups')
    if label_col:
        mask_all = df[label_col].astype(str).str.lower().str.contains("all serogroups", na=False)
        if mask_all.any():
            df = df[mask_all].copy()

    # 3. Clean strings & filter jurisdiction
    df["_area_upper"] = df[area_col].astype(str).str.strip().str.upper()
    code = get_state_code(jurisdiction)

    if code == "US":
        us_mask = df["_area_upper"].isin(["US RESIDENTS", "U.S. RESIDENTS", "TOTAL", "UNITED STATES", "US TOTAL", "TOTAL 50 STATES AND DC"])
        if us_mask.any():
            sub = df[us_mask].copy()
        else:
            divisions = [
                "NEW ENGLAND", "MIDDLE ATLANTIC", "EAST NORTH CENTRAL", "WEST NORTH CENTRAL",
                "SOUTH ATLANTIC", "EAST SOUTH CENTRAL", "WEST SOUTH CENTRAL", "MOUNTAIN", "PACIFIC",
                "TERRITORIES", "AMERICAN SAMOA", "GUAM", "COMMONWEALTH OF NORTHERN MARIANA ISLANDS",
                "PUERTO RICO", "VIRGIN ISLANDS"
            ]
            sub = df[~df["_area_upper"].isin(divisions)].copy()
    else:
        sub = df[df["_area_upper"].isin(["MI", "MICHIGAN", "MICH."]) | df["_area_upper"].str.startswith("MICH")].copy()

    if sub.empty:
        return pd.DataFrame()

    # 4. Clean numeric case values
    cleaned = (
        sub[case_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace(["-", "N", "U", "NP", "NC", "NN", "null", "nan", "None", "", "."], "0")
    )
    sub["cases"] = pd.to_numeric(cleaned, errors="coerce").fillna(0.0)

    sub["year_num"] = pd.to_numeric(sub[year_col], errors="coerce")
    sub["week_num"] = pd.to_numeric(sub[week_col], errors="coerce")
    sub = sub.dropna(subset=["year_num", "week_num"]).copy()
    sub["year_num"] = sub["year_num"].astype(int)
    sub["week_num"] = sub["week_num"].astype(int)
    sub = sub[(sub["week_num"] >= 1) & (sub["week_num"] <= 53)]

    # 5. Group by Year + Week and create date index
    agg = sub.groupby(["year_num", "week_num"])["cases"].sum().reset_index()
    agg["Date"] = pd.to_datetime(agg["year_num"].astype(str) + "-01-01") + pd.to_timedelta((agg["week_num"] - 1) * 7, unit="D")
    
    res = agg.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    # 6. Compute 4-week rolling average for clean trend visualization
    res["4-Week Average"] = res["cases"].rolling(window=4, min_periods=1).mean().round(1)
    res = res.rename(columns={"cases": "Weekly Cases (Raw)"})

    return res[["Weekly Cases (Raw)", "4-Week Average"]]