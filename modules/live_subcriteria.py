"""
live_subcriteria.py

The live subcriteria computed from the surveillance feeds at the same
completed week the rest of the dashboard uses, and the history behind the
historical level of the limits hierarchy.

Admission: a live subcriterion enters an agent only when at least two
diseases hold a value (coverage rule); otherwise it is reported as not
admitted with its reason. Meningococcal disease is not covered by the
respiratory feeds, so its live cells are missing and are excluded cell by
cell with weight renormalization.

Freshness: a live value whose date is more than MAX_AGE_WEEKS older than
the dashboard's reference week is not current and is treated as missing,
with the reason recorded (freshness_report).

History window (limits and scenario tightening): from the end of the
federal COVID 19 public health emergency (11 May 2023), growing until it
holds five complete respiratory seasons (week 40 to week 39), then rolling
so that it always holds the five most recent complete seasons plus the
current one. The full history remains selectable.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules import live_connectors as lc
from modules import feeds

_MEMO: Dict[tuple, object] = {}


def _memo(fn):
    """Memoizes on the arguments, the feed files and the replay date."""
    def wrapper(*args, **kwargs):
        key = (fn.__name__, args, tuple(sorted(kwargs.items())),
               feeds.feed_signature(), str(lc.AS_OF))
        if key not in _MEMO:
            if len(_MEMO) > 512:
                _MEMO.clear()
            _MEMO[key] = fn(*args, **kwargs)
        return _MEMO[key]
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper

MIN_COVERAGE = 2
PEAK_WINDOW = 52          # weeks for the trailing peak
VELOCITY_LAG = 2          # weeks for surge velocity
MAX_AGE_WEEKS = 4         # freshness rule for live values
PHE_END = pd.Timestamp("2023-05-11")
ROLLING_SEASONS = 5

WINDOWS = {
    "post_phe_rolling5": "Since the end of the public health emergency (11 May 2023), "
                         "then the five most recent complete seasons",
    "full": "Full history of each series",
}
DEFAULT_WINDOW = "post_phe_rolling5"

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
SOURCE_OF = {name: key for items in LIVE.values() for name, key in items}

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


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def season_start(ts: pd.Timestamp) -> pd.Timestamp:
    """Start (1 October) of the respiratory season containing ts."""
    y = ts.year if ts.month >= 10 else ts.year - 1
    return pd.Timestamp(year=y, month=10, day=1)


def window_start(ref: pd.Timestamp, window: str = DEFAULT_WINDOW) -> Optional[pd.Timestamp]:
    if window == "full" or ref is None or pd.isna(ref):
        return None
    current = season_start(ref)
    rolling = pd.Timestamp(year=current.year - ROLLING_SEASONS, month=10, day=1)
    return max(PHE_END, rolling)


def _upto_target(df: pd.DataFrame, date_col: str, lag: int) -> pd.DataFrame:
    ts, latest = lc.extract_completed_series(df, date_col, lag)
    if latest is None or ts is None or ts.empty:
        return pd.DataFrame()
    return ts.sort_values(date_col)


def _series(jurisdiction: str, lag: int) -> Dict[str, Dict[str, pd.Series]]:
    """source key -> disease -> weekly series (date index) up to the target week."""
    out: Dict[str, Dict[str, pd.Series]] = {k: {} for k in ["adm", "icu", "ped", "ger", "ed", "pos"]}
    h = _upto_target(lc._nhsn_jur(jurisdiction), "weekendingdate", lag)
    for d, (stem, _, _) in RESP.items():
        if h.empty:
            break
        idx = pd.to_datetime(h["weekendingdate"])
        for key, col in [("adm", f"totalconf{stem}newadmper100k"),
                         ("icu", f"pctconf{stem}hosppatsicu"),
                         ("ped", f"numconf{stem}newadmped0to4per100k"),
                         ("ger", f"numconf{stem}newadmadult75plusper100k")]:
            if col in h.columns:
                out[key][d] = pd.Series(_num(h[col]).values, index=idx.values).dropna()
    e = _upto_target(lc._nssp_jur(jurisdiction), "week_end", lag)
    for d, (_, col, _) in RESP.items():
        if not e.empty and col in e.columns:
            out["ed"][d] = pd.Series(_num(e[col]).values,
                                     index=pd.to_datetime(e["week_end"]).values).dropna()
    if lc.is_national(jurisdiction):
        raw = lc.load_raw_sheet("Percent of Tests Positive for V")
        if not raw.empty and "pathogen" in raw.columns:
            for d, (_, _, keys) in RESP.items():
                m = raw["pathogen"].astype(str).str.lower().apply(lambda x: any(k in x for k in keys))
                p = _upto_target(raw[m], "week_end", lag)
                if not p.empty:
                    g = p.groupby("week_end")["percent_test_positivity"].apply(lambda s: _num(s).median())
                    g.index = pd.to_datetime(g.index)
                    out["pos"][d] = g.sort_index().dropna()
    return out


def _derived(s: pd.Series):
    s = s.dropna().sort_index()
    peak = s.rolling(PEAK_WINDOW, min_periods=max(4, PEAK_WINDOW // 2)).max()
    base = s.shift(VELOCITY_LAG)
    vel = ((s - base) / base * 100.0).where(base > 0)
    return peak, vel.replace([np.inf, -np.inf], np.nan)


@_memo
def criterion_series(jurisdiction: str = "National", lag: int = 1) -> Dict[str, Dict[str, pd.Series]]:
    """criterion -> disease -> weekly series of the live criterion itself."""
    ser = _series(jurisdiction, lag)
    out: Dict[str, Dict[str, pd.Series]] = {}
    for name, key in SOURCE_OF.items():
        out[name] = {}
        for d in RESP:
            s = ser["adm" if key in ("peak", "vel") else key].get(d)
            if s is None or s.empty:
                continue
            if key in ("peak", "vel"):
                pk, vl = _derived(s)
                s = pk if key == "peak" else vl
            out[name][d] = s.dropna()
    return out


@_memo
def reference_week(jurisdiction: str = "National", lag: int = 1) -> Optional[pd.Timestamp]:
    """The dashboard's target week: the latest completed NHSN week minus the
    lag (the anchor of freshness and of the history window)."""
    ts, latest = lc.extract_completed_series(lc._nhsn_jur(jurisdiction), "weekendingdate", lag)
    if latest is None:
        return None
    return pd.Timestamp(latest["weekendingdate"])


def live_panel(jurisdiction: str = "National", lag: int = 1) -> pd.DataFrame:
    """Disease x live criterion values at the target week; NaN where the
    disease is not covered or the latest value is stale."""
    return _panel(jurisdiction, lag)[0]


def freshness_report(jurisdiction: str = "National", lag: int = 1) -> pd.DataFrame:
    return _panel(jurisdiction, lag)[1]


@_memo
def _panel(jurisdiction: str, lag: int):
    cs = criterion_series(jurisdiction, lag)
    ref = reference_week(jurisdiction, lag)
    rows, notes = {}, []
    for d in list(RESP) + ["Meningococcal Disease (Meningitis)"]:
        r = {}
        for name in SOURCE_OF:
            s = cs.get(name, {}).get(d)
            v = np.nan
            if s is not None and len(s):
                last_date = pd.Timestamp(s.index.max())
                age = (ref - last_date).days / 7.0 if ref is not None else 0.0
                if age > MAX_AGE_WEEKS:
                    notes.append({"Disease": d, "Criterion": name, "Latest date": str(last_date)[:10],
                                  "Age (weeks)": round(age, 1), "Status": "stale, treated as missing"})
                else:
                    v = float(s.iloc[-1])
            r[name] = v
        rows[d] = r
    if not lc.is_national(jurisdiction):
        for d in RESP:
            notes.append({"Disease": d, "Criterion": "Current Test Positivity (%)",
                          "Latest date": "", "Age (weeks)": None,
                          "Status": "NREVSS positivity is published nationally only; missing in a state scope"})
    return pd.DataFrame(rows).T, pd.DataFrame(notes)


def historical_values(jurisdiction: str = "National", lag: int = 1,
                      window: str = DEFAULT_WINDOW) -> Dict[str, pd.Series]:
    """criterion -> pooled history (covered diseases) inside the window."""
    cs = criterion_series(jurisdiction, lag)
    ref = reference_week(jurisdiction, lag)
    start = window_start(ref, window)
    out = {}
    for name, per in cs.items():
        parts = [s[s.index >= start] if start is not None else s for s in per.values()]
        parts = [p for p in parts if len(p)]
        out[name] = pd.concat(parts) if parts else pd.Series(dtype=float)
    return out


@_memo
def historical_stats(jurisdiction: str = "National", lag: int = 1,
                     window: str = DEFAULT_WINDOW) -> Dict[str, Dict[str, float]]:
    """criterion -> quartiles, P90, n and the window, from the pooled history."""
    hv = historical_values(jurisdiction, lag, window)
    ref = reference_week(jurisdiction, lag)
    start = window_start(ref, window)
    out = {}
    for name, v in hv.items():
        if len(v) >= 8:
            out[name] = {"p25": float(v.quantile(0.25)), "p50": float(v.median()),
                         "p75": float(v.quantile(0.75)), "p90": float(v.quantile(0.90)),
                         "n": int(len(v)),
                         "window_start": str(start)[:10] if start is not None else "full",
                         "reference_week": str(ref)[:10] if ref is not None else ""}
    return out


def attach(agent_key: str, profile: pd.DataFrame, jurisdiction: str = "National",
           lag: int = 1) -> pd.DataFrame:
    names = criteria_for(agent_key)
    if not names:
        return profile
    panel = live_panel(jurisdiction, lag)
    out = profile.copy()
    for n in names:
        out[n] = out["Disease Type"].map(panel[n].to_dict()) if n in panel.columns else np.nan
    return out


def active_criteria(profile: pd.DataFrame, criteria: List[str]) -> List[str]:
    """Coverage rule: active when at least MIN_COVERAGE diseases hold a value."""
    return [c for c in criteria
            if c in profile.columns and _num(profile[c]).notna().sum() >= MIN_COVERAGE]
