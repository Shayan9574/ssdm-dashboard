"""
live_connectors.py

Readers over the CDC surveillance feeds (modules.feeds). Every reader
returns the value for the dashboard's target week, the latest completed
week minus the reporting lag, together with its date, and never fills a
missing value.

AS_OF (optional Timestamp) truncates every feed at a past date, which is
how the weekly replay reconstructs the dashboard as it stood that week.

Scope rules: NREVSS test positivity is published nationally only, so in a
state scope it is missing rather than silently national; the wastewater
feed is collected for Michigan sewersheds only and is labelled as such in
every scope.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from modules import feeds

AS_OF: Optional[pd.Timestamp] = None

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
NATIONAL_NAMES = {"national", "us", "usa", "united states", "all"}

RESP = {"COVID-19": "c19", "Influenza": "flu", "RSV": "rsv"}


def is_national(jur: str) -> bool:
    return str(jur).strip().lower() in NATIONAL_NAMES


def get_state_code(jur: str) -> str:
    j = str(jur).strip().lower()
    if j in NATIONAL_NAMES:
        return "US"
    return NAME_TO_CODE.get(j, jur.upper()[:2])


def get_state_full(jur: str) -> str:
    code = get_state_code(jur)
    if code == "US":
        return "United States"
    return STATE_NAMES.get(code, jur.title())


def load_raw_sheet(sheet_name: str) -> pd.DataFrame:
    return feeds.load_feed(sheet_name)


def _truncate(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    if AS_OF is not None:
        out = out[out[date_col] <= AS_OF]
    return out


def extract_completed_series(df: pd.DataFrame, date_col: str,
                             lag_complete_weeks: int = 1) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Weekly feeds: one row per week; the target row is the latest week
    minus the lag."""
    if df.empty or date_col not in df.columns:
        return df, None
    d = _truncate(df, date_col).sort_values(date_col, kind="mergesort").reset_index(drop=True)
    if d.empty:
        return d, None
    weeks = d[date_col].drop_duplicates().sort_values().reset_index(drop=True)
    target = weeks.iloc[max(0, len(weeks) - 1 - lag_complete_weeks)]
    rows = d[d[date_col] == target]
    return d[d[date_col] <= target], rows.iloc[-1]


# --- NHSN hospital respiratory data ----------------------------------------

def _nhsn_jur(jurisdiction: str) -> pd.DataFrame:
    raw = load_raw_sheet("Weekly Hospital Respiratory Dat")
    if raw.empty or "jurisdiction" not in raw.columns:
        return pd.DataFrame()
    code = get_state_code(jurisdiction)
    j = raw["jurisdiction"].astype(str).str.upper()
    return raw[j.isin(["USA", "US", "NATIONAL"])] if code == "US" else raw[j == code]


def get_hospital_respiratory_surveillance(jurisdiction: str = "National",
                                          lag_complete_weeks: int = 1) -> Dict[str, Any]:
    ts, latest = extract_completed_series(_nhsn_jur(jurisdiction), "weekendingdate", lag_complete_weeks)
    if latest is None:
        return {}
    g = lambda c: pd.to_numeric(latest.get(c), errors="coerce")
    out = {"as_of_date": str(latest.get("weekendingdate"))[:10],
           "inpatient_occ": g("pctinptbedsocc"), "icu_occ": g("pcticubedsocc")}
    for d, s in [("covid", "c19"), ("flu", "flu"), ("rsv", "rsv")]:
        out[f"{d}_adm_per100k"] = g(f"totalconf{s}newadmper100k")
        out[f"{d}_icu_rate"] = g(f"pctconf{s}hosppatsicu")
    return out


def _nhsn_trend(jurisdiction: str, cols: Dict[str, str]) -> pd.DataFrame:
    sub = _truncate(_nhsn_jur(jurisdiction), "weekendingdate")
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values("weekendingdate").set_index("weekendingdate")
    sub.index.name = "Date"
    return pd.DataFrame({k: pd.to_numeric(sub[c], errors="coerce") for k, c in cols.items()
                         if c in sub.columns}).dropna(how="all")


def get_hospital_admission_trends(jurisdiction: str = "National") -> pd.DataFrame:
    return _nhsn_trend(jurisdiction, {"COVID-19": "totalconfc19newadmper100k",
                                      "Influenza": "totalconfflunewadmper100k",
                                      "RSV": "totalconfrsvnewadmper100k"})


def get_hospital_occupancy_trends(jurisdiction: str = "National") -> pd.DataFrame:
    return _nhsn_trend(jurisdiction, {"Inpatient Bed Occupancy (%)": "pctinptbedsocc",
                                      "ICU Bed Occupancy (%)": "pcticubedsocc"})


# --- NSSP emergency department visits --------------------------------------

def _nssp_jur(jurisdiction: str) -> pd.DataFrame:
    raw = load_raw_sheet("NSSP Emergency Department Visit")
    if raw.empty or "geography" not in raw.columns:
        return pd.DataFrame()
    geo = raw["geography"].astype(str).str.lower()
    cty = raw["county"].astype(str) if "county" in raw.columns else "All"
    if is_national(jurisdiction):
        return raw[geo.isin(["united states", "national", "us"]) & (cty == "All")]
    return raw[(geo == get_state_full(jurisdiction).lower()) & (cty == "All")]


def get_ed_visit_surveillance(jurisdiction: str = "National",
                              lag_complete_weeks: int = 1) -> Dict[str, Any]:
    ts, latest = extract_completed_series(_nssp_jur(jurisdiction), "week_end", lag_complete_weeks)
    if latest is None:
        return {}
    g = lambda c: pd.to_numeric(latest.get(c), errors="coerce")
    return {"covid_ed_pct": g("percent_visits_covid"), "flu_ed_pct": g("percent_visits_influenza"),
            "rsv_ed_pct": g("percent_visits_rsv"), "as_of_date": str(latest.get("week_end"))[:10]}


def get_ed_visit_trends(jurisdiction: str = "National") -> pd.DataFrame:
    sub = _truncate(_nssp_jur(jurisdiction), "week_end")
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values("week_end").set_index("week_end")
    sub.index.name = "Date"
    return pd.DataFrame({"COVID-19": pd.to_numeric(sub["percent_visits_covid"], errors="coerce"),
                         "Influenza": pd.to_numeric(sub["percent_visits_influenza"], errors="coerce"),
                         "RSV": pd.to_numeric(sub["percent_visits_rsv"], errors="coerce")}).dropna(how="all")


# --- NREVSS test positivity (national only) --------------------------------

def get_test_positivity_trends(jurisdiction: str = "National") -> pd.DataFrame:
    if not is_national(jurisdiction):
        return pd.DataFrame()
    raw = _truncate(load_raw_sheet("Percent of Tests Positive for V"), "week_end")
    if raw.empty:
        return pd.DataFrame()
    piv = raw.pivot_table(index="week_end", columns="pathogen",
                          values="percent_test_positivity", aggfunc="median")
    piv.index.name = "Date"
    return piv.sort_index().dropna(how="all")


# --- CFA epidemic trends and Rt --------------------------------------------

def _rt_jur(jurisdiction: str) -> pd.DataFrame:
    raw = load_raw_sheet("CDC Epidemic Trends and Rt")
    if raw.empty or "state" not in raw.columns:
        return pd.DataFrame()
    st_ = raw["state"].astype(str).str.lower()
    sub = raw[st_.isin(["united states", "national", "us"])] if is_national(jurisdiction) \
        else raw[st_ == get_state_full(jurisdiction).lower()]
    sub = _truncate(sub, "date")
    if "as_of" in sub.columns:
        sub = sub.copy()
        sub["as_of"] = pd.to_datetime(sub["as_of"], errors="coerce")
        if AS_OF is not None:
            sub = sub[sub["as_of"] <= AS_OF]
    return sub


def _latest_run_daily(sub: pd.DataFrame) -> pd.DataFrame:
    """Daily estimates superseded by later model runs: for every date the
    estimate of the most recent run (as_of) that covers it."""
    if "as_of" not in sub.columns:
        return sub.sort_values("date")
    s = sub.sort_values(["date", "as_of"], kind="mergesort")
    return s.groupby("date", as_index=False).tail(1).sort_values("date")


def get_rt_surveillance(jurisdiction: str = "National",
                        lag_complete_weeks: int = 1) -> Dict[str, Any]:
    """Per disease: the median of the daily Rt and growth probability over
    the seven days ending at the target date (latest estimated date minus
    the lag in weeks), taken from the most recent model run of each date."""
    sub = _rt_jur(jurisdiction)
    out = {}
    if sub.empty:
        return out
    for disease in sub["disease"].dropna().unique():
        d = _latest_run_daily(sub[sub["disease"] == disease])
        if d.empty:
            continue
        target = d["date"].max() - pd.Timedelta(days=7 * lag_complete_weeks)
        win = d[(d["date"] > target - pd.Timedelta(days=7)) & (d["date"] <= target)]
        if win.empty:
            continue
        out[str(disease).strip()] = {
            "median": float(pd.to_numeric(win["median"], errors="coerce").median()),
            "p_growing": float(pd.to_numeric(win["p_growing"], errors="coerce").median()),
            "as_of_date": str(target)[:10],
        }
    return out


def get_rt_trends(jurisdiction: str = "National") -> pd.DataFrame:
    sub = _rt_jur(jurisdiction)
    if sub.empty:
        return pd.DataFrame()
    parts = {}
    for disease in sub["disease"].dropna().unique():
        d = _latest_run_daily(sub[sub["disease"] == disease])
        parts[str(disease)] = pd.to_numeric(d.set_index("date")["median"], errors="coerce")
    df = pd.DataFrame(parts)
    df.index.name = "Date"
    return df.sort_index().dropna(how="all")


# --- NWSS wastewater (Michigan sewersheds only) ----------------------------

WASTEWATER_SCOPE = "Michigan sewersheds"


def _ww() -> pd.DataFrame:
    raw = load_raw_sheet("CDC Wastewater Data for SARS-Co")
    if raw.empty or "state_territory" not in raw.columns:
        return pd.DataFrame()
    sub = raw[raw["state_territory"].astype(str).str.strip().str.lower().isin(["mi", "michigan"])]
    return _truncate(sub, "sample_collect_date")


def get_wastewater_surveillance(jurisdiction: str = "National",
                                lag_complete_weeks: int = 1) -> Dict[str, Any]:
    """Median sewershed concentration over the week ending at the target
    date; always the Michigan sewersheds, whatever the scope."""
    sub = _ww()
    if sub.empty:
        return {}
    last = sub["sample_collect_date"].max() - pd.Timedelta(days=7 * lag_complete_weeks)
    win = sub[(sub["sample_collect_date"] > last - pd.Timedelta(days=7))
              & (sub["sample_collect_date"] <= last)]
    v = pd.to_numeric(win["pcr_target_avg_conc_lin"], errors="coerce").median()
    return {"viral_conc": float(v) if v == v else np.nan, "as_of_date": str(last)[:10],
            "scope": WASTEWATER_SCOPE}


def get_wastewater_trends(jurisdiction: str = "National") -> pd.DataFrame:
    sub = _ww()
    if sub.empty:
        return pd.DataFrame()
    daily = sub.groupby("sample_collect_date")["pcr_target_avg_conc_lin"].apply(
        lambda s: pd.to_numeric(s, errors="coerce").median())
    out = daily.rolling(7, min_periods=1).mean().to_frame(
        name=f"SARS-CoV-2 Viral Load, {WASTEWATER_SCOPE} (copies/L)")
    out.index.name = "Date"
    return out.sort_index().dropna(how="all")


# --- NNDSS meningococcal disease -------------------------------------------

def _mmwr_week_end(year: int, week: int) -> pd.Timestamp:
    """Saturday ending MMWR week `week` of `year` (MMWR week 1 is the first
    week, Sunday to Saturday, with at least four days in the year)."""
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    # Sunday starting week 1
    start = jan1 - pd.Timedelta(days=(jan1.dayofweek + 1) % 7)
    if (jan1.dayofweek + 1) % 7 > 3:
        start += pd.Timedelta(days=7)
    return start + pd.Timedelta(days=7 * (week - 1) + 6)


def nndss_weekly(df: pd.DataFrame, jurisdiction: str = "National") -> pd.DataFrame:
    """year, week, cases for the scope. National: the US residents row
    (labelled 'US RESIDENTS' to 2024 and 'U.S. RESIDENTS' from 2025); the
    'TOTAL' row, which adds non residents and territories, is used only for
    a week without a residents row. Unreported cells stay missing."""
    area = df["states"].astype(str).str.strip().str.upper()
    if get_state_code(jurisdiction) == "US":
        pri = area.map({"US RESIDENTS": 0, "U.S. RESIDENTS": 0, "TOTAL": 1})
    else:
        pri = area.map({"MICHIGAN": 0, "MI": 0})
    sub = df[pri.notna()].assign(_p=pri[pri.notna()])
    if sub.empty:
        return pd.DataFrame()
    cases = pd.to_numeric(sub["m1"].astype(str).str.replace(",", "", regex=False).str.strip(),
                          errors="coerce")
    y = pd.to_numeric(sub["year"], errors="coerce")
    w = pd.to_numeric(sub["week"], errors="coerce")
    ok = y.notna() & w.notna() & (w >= 1) & (w <= 53)
    t = pd.DataFrame({"year": y[ok].astype(int), "week": w[ok].astype(int),
                      "cases": cases[ok], "_p": sub["_p"][ok]})
    t = t.sort_values(["year", "week", "_p"]).groupby(["year", "week"], as_index=False).first()
    return t[["year", "week", "cases"]]


def get_meningitis_trends(jurisdiction: str = "National") -> pd.DataFrame:
    """Weekly meningococcal cases (all serogroups) for the scope, with a four
    week moving average; unreported cells stay missing."""
    raw = load_raw_sheet("NNDSS Weekly Data")
    if raw.empty or not {"year", "week", "states", "m1"} <= set(raw.columns):
        return pd.DataFrame()
    df = raw.copy()
    if "label" in df.columns:
        m = df["label"].astype(str).str.lower().str.contains("all serogroups", na=False)
        if m.any():
            df = df[m]
    t = nndss_weekly(df, jurisdiction)
    if t.empty:
        return pd.DataFrame()
    t["Date"] = [_mmwr_week_end(a, b) for a, b in zip(t["year"], t["week"])]
    if AS_OF is not None:
        t = t[t["Date"] <= AS_OF]
    res = t.sort_values("Date").set_index("Date")[["cases"]]
    res["4-Week Average"] = res["cases"].rolling(window=4, min_periods=1).mean().round(1)
    return res.rename(columns={"cases": "Weekly Cases"})
