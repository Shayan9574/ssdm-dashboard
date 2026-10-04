"""
live_subcriteria.py

Agents L and H (Section 4.2): the live and hybrid subcriteria named in the
Data Dictionary, computed from the surveillance feeds at the same completed
week the rest of the dashboard uses (extract_completed_series, so the
weekly replay and every lag setting apply unchanged).

Every live subcriterion is a cost (positive) criterion: higher means greater
concern. A live subcriterion is ADMITTED into an agent only when at least
MIN_COVERAGE of the four diseases hold a value (coverage rule, Section 4.4);
otherwise it is reported as not admitted with its reason. Meningococcal
disease is not covered by the respiratory feeds, so its live cells are
missing and are excluded cell by cell with weight renormalization.

historical_stats() returns, for every live subcriterion, the quartiles of
its own history pooled over the covered diseases up to the same completed
week; the median is the historical level of the limits hierarchy
(Section 4.5.3) and the quartiles serve scenario limit overrides.
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from modules import live_connectors as lc

MIN_COVERAGE = 2
PEAK_WINDOW = 52          # weeks for the trailing peak
VELOCITY_LAG = 2          # weeks for surge velocity

RESP = {  # disease -> NHSN column stem, NSSP column, NREVSS pathogen matcher
    "COVID-19": ("c19", "percent_visits_covid", ("covid",)),
    "Influenza": ("flu", "percent_visits_influenza", ("influenza", "flu")),
    "Respiratory Syncytial Virus (RSV)": ("rsv", "percent_visits_rsv", ("rsv", "syncytial")),
}

# agent -> list of (criterion name, source key)
LIVE: Dict[str, List[Tuple[str, str]]] = {
    "A1": [("Current Weekly Hospitalization Rate (per 100k)", "adm"),
           ("Peak Weekly Hospitalization Rate, 52 Weeks (per 100k)", "peak"),
           ("Current ICU Share of Hospitalized (%)", "icu")],
    "A2": [("Current Test Positivity (%)", "pos")],
    "A3": [("Current ED Visit Share (%)", "ed"),
           ("Hospitalization Surge Velocity (%)", "vel")],
    "A6": [("Current Pediatric Hospitalization Rate <5 (per 100k)", "ped"),
           ("Current Geriatric Hospitalization Rate 75+ (per 100k)", "ger")],
}

NOT_ADMITTED = [
    ("A1", "Peak Weekly Incidence Rate (NNDSS)",
     "notifiable counts cover Meningococcal disease only (coverage below two diseases)"),
    ("A3", "Peak Hospital Bed Occupancy (%)",
     "system level measure, identical for every disease, carries no discrimination"),
    ("A3", "Peak ICU Surge Pressure (%)",
     "system level measure, identical for every disease, carries no discrimination"),
    ("A7", "Wastewater Viral Concentration",
     "SARS-CoV-2 only and Michigan sewersheds only (coverage below two diseases)"),
]


def criteria_for(agent_key: str) -> List[str]:
    return [c for c, _ in LIVE.get(agent_key, [])]


def _nhsn(jurisdiction: str):
    raw = lc.load_raw_sheet("Weekly Hospital Respiratory Dat")
    code = lc.get_state_code(jurisdiction)
    if code == "US":
        return raw[raw["jurisdiction"].astype(str).str.upper().isin(["USA", "US", "NATIONAL"])]
    return raw[raw["jurisdiction"].astype(str).str.upper() == code]


def _upto_target(df: pd.DataFrame, date_col: str, lag: int) -> pd.DataFrame:
    """Rows up to and including the completed week the dashboard uses."""
    ts, latest = lc.extract_completed_series(df, date_col, lag)
    if latest is None or ts is None or ts.empty:
        return pd.DataFrame()
    ts = ts.copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    return ts[ts[date_col] <= pd.to_datetime(latest[date_col])].sort_values(date_col)


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _series(jurisdiction: str, lag: int) -> Dict[str, Dict[str, pd.Series]]:
    """source key -> disease -> weekly series up to the target week."""
    out: Dict[str, Dict[str, pd.Series]] = {k: {} for k in
                                            ["adm", "icu", "ped", "ger", "ed", "pos"]}
    try:
        h = _upto_target(_nhsn(jurisdiction), "weekendingdate", lag)
    except Exception:
        h = pd.DataFrame()
    for d, (stem, _, _) in RESP.items():
        if h.empty:
            continue
        idx = h["weekendingdate"]
        for key, col in [("adm", f"totalconf{stem}newadmper100k"),
                         ("icu", f"pctconf{stem}hosppatsicu"),
                         ("ped", f"numconf{stem}newadmped0to4per100k"),
                         ("ger", f"numconf{stem}newadmadult75plusper100k")]:
            if col in h.columns:
                out[key][d] = pd.Series(_num(h[col]).values, index=idx.values)
    try:
        raw = lc.load_raw_sheet("NSSP Emergency Department Visit")
        if jurisdiction.lower() in {"national", "us", "usa", "united states"}:
            ej = raw[raw["geography"].astype(str).str.lower().isin(["united states", "national", "us"])
                     & (raw["county"] == "All")]
        else:
            ej = raw[(raw["geography"].astype(str).str.lower() == lc.get_state_full(jurisdiction).lower())
                     & (raw["county"] == "All")]
        e = _upto_target(ej, "week_end", lag)
        for d, (_, col, _) in RESP.items():
            if not e.empty and col in e.columns:
                out["ed"][d] = pd.Series(_num(e[col]).values, index=e["week_end"].values)
    except Exception:
        pass
    try:
        raw = lc.load_raw_sheet("Percent of Tests Positive for V")
        for d, (_, _, keys) in RESP.items():
            m = raw["pathogen"].astype(str).str.lower().apply(lambda x: any(k in x for k in keys))
            p = _upto_target(raw[m], "week_end", lag)
            if not p.empty:
                g = p.groupby("week_end")["percent_test_positivity"].apply(
                    lambda s: _num(s).mean())
                out["pos"][d] = g.sort_index()
    except Exception:
        pass
    return out


def _derived(s: pd.Series):
    s = s.dropna()
    peak = s.rolling(PEAK_WINDOW, min_periods=max(4, PEAK_WINDOW // 2)).max()
    base = s.shift(VELOCITY_LAG)
    vel = ((s - base) / base * 100.0).where(base > 0)
    return peak, vel


def live_panel(jurisdiction: str = "National", lag: int = 1) -> pd.DataFrame:
    """Disease x live criterion values at the target week (NaN where uncovered)."""
    ser = _series(jurisdiction, lag)
    rows = {}
    for d in list(RESP) + ["Meningococcal Disease (Meningitis)"]:
        r = {}
        for a, items in LIVE.items():
            for name, key in items:
                v = np.nan
                if key in ("peak", "vel"):
                    s = ser["adm"].get(d)
                    if s is not None and len(s.dropna()):
                        pk, vl = _derived(s)
                        src = pk if key == "peak" else vl
                        v = src.iloc[-1] if len(src) else np.nan
                else:
                    s = ser[key].get(d)
                    if s is not None and len(s.dropna()):
                        v = s.dropna().iloc[-1]
                r[name] = float(v) if v == v and v is not None else np.nan
        rows[d] = r
    return pd.DataFrame(rows).T


def historical_stats(jurisdiction: str = "National", lag: int = 1) -> Dict[str, Dict[str, float]]:
    """criterion -> {p25, p50, p75, n} over its own history to the target week,
    pooled across the covered diseases."""
    ser = _series(jurisdiction, lag)
    out = {}
    for a, items in LIVE.items():
        for name, key in items:
            vals = []
            for d in RESP:
                s = ser["adm" if key in ("peak", "vel") else key].get(d)
                if s is None:
                    continue
                if key in ("peak", "vel"):
                    pk, vl = _derived(s)
                    s = pk if key == "peak" else vl
                vals.append(s.replace([np.inf, -np.inf], np.nan).dropna())
            v = pd.concat(vals) if vals else pd.Series(dtype=float)
            if len(v) >= 8:
                out[name] = {"p25": float(v.quantile(0.25)), "p50": float(v.median()),
                             "p75": float(v.quantile(0.75)), "n": int(len(v))}
    return out


def attach(agent_key: str, profile: pd.DataFrame, jurisdiction: str = "National",
           lag: int = 1) -> pd.DataFrame:
    """Adds the agent's live columns to its baseline profile."""
    names = criteria_for(agent_key)
    if not names:
        return profile
    panel = live_panel(jurisdiction, lag)
    out = profile.copy()
    for n in names:
        out[n] = out["Disease Type"].map(panel[n].to_dict()) if n in panel.columns else np.nan
    return out


def active_criteria(profile: pd.DataFrame, criteria: List[str]) -> List[str]:
    """Coverage rule: a criterion is active when at least MIN_COVERAGE
    diseases hold a numeric value."""
    act = []
    for c in criteria:
        if c in profile.columns and pd.to_numeric(profile[c], errors="coerce").notna().sum() >= MIN_COVERAGE:
            act.append(c)
    return act
