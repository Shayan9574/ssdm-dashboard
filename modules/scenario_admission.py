"""
scenario_admission.py

Data driven admission of the scenario registry (thesis Chapter 2, "Admission
from Documented History"; paper Section 4.6). For every scenario with a
countable marker condition the weekly marker series is rebuilt from the CDC
feeds, the weeks satisfying the marker are identified, contiguous runs of
such weeks are counted as episodes, and the observable window Y_s is the
span over which the marker can be evaluated. Event catalog scenarios (no
weekly series) keep their curated, documented counts.

The same series give the two data driven scenario inputs:
  * the marker threshold, used as the scenario's acceptable limit override
    (tightening): the P90 of the criterion's history, or the absolute
    threshold of an absolute marker (Rt = 1);
  * the emphasis multiplier of each emphasized criterion: its median during
    the scenario's episode weeks divided by its median over the scenario's
    observable window, never below one (Eq. scenario emphasis).

All thresholds use the full observable span of each marker series, because
an occurrence of a condition is defined against the whole record of the
series that marks it.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules import live_connectors as lc

MARKER_Q = 0.90
NAT = "National"


# ----------------------------------------------------------------------
# weekly series (national scope, week ending Saturday)
# ----------------------------------------------------------------------

def _weekly(s: pd.Series, how: str = "median") -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    s.index = pd.to_datetime(s.index)
    r = s.resample("W-SAT")
    return (r.median() if how == "median" else r.max()).dropna()


def nhsn_admissions() -> Dict[str, pd.Series]:
    t = lc.get_hospital_admission_trends(NAT)
    return {c: t[c].dropna() for c in t.columns} if not t.empty else {}


def nssp_ed() -> Dict[str, pd.Series]:
    t = lc.get_ed_visit_trends(NAT)
    return {c: t[c].dropna() for c in t.columns} if not t.empty else {}


def nrevss_positivity() -> Dict[str, pd.Series]:
    t = lc.get_test_positivity_trends(NAT)
    return {str(c): t[c].dropna() for c in t.columns} if not t.empty else {}


def rt_weekly() -> Dict[str, pd.Series]:
    t = lc.get_rt_trends(NAT)
    return {str(c): _weekly(t[c]) for c in t.columns} if not t.empty else {}


def wastewater_weekly() -> pd.Series:
    t = lc.get_wastewater_trends(NAT)
    if t.empty:
        return pd.Series(dtype=float)
    return _weekly(t.iloc[:, 0])


def respnet_65() -> pd.Series:
    raw = lc.load_raw_sheet("RESP-NET Rates and Clinical Dat")
    if raw.empty:
        return pd.Series(dtype=float)
    m = ((raw["surveillance_network"].astype(str) == "Combined")
         & (raw["age_category"].astype(str) == "65+ yr")
         & (raw["data_type"].astype(str) == "Weekly Rate")
         & (raw["sex"].astype(str) == "All") & (raw["race"].astype(str) == "All")
         & (raw["state"].astype(str) == "Overall"))
    sub = raw[m].copy()
    if "rate_type" in sub.columns and (sub["rate_type"] == "Observed").any():
        sub = sub[sub["rate_type"] == "Observed"]
    sub = lc._truncate(sub, "date")
    s = pd.Series(pd.to_numeric(sub["estimate"], errors="coerce").values,
                  index=pd.to_datetime(sub["date"]).values)
    return s.groupby(level=0).median().sort_index().dropna()


def nndss_mening() -> pd.Series:
    raw = lc.load_raw_sheet("NNDSS Weekly Data")
    if raw.empty:
        return pd.Series(dtype=float)
    df = raw[raw["label"].astype(str).str.lower().str.contains("all serogroups", na=False)]
    t = lc.nndss_weekly(df, NAT)
    if t.empty:
        return pd.Series(dtype=float)
    idx = [lc._mmwr_week_end(a, b) for a, b in zip(t["year"], t["week"])]
    s = pd.Series(t["cases"].values, index=pd.DatetimeIndex(idx)).sort_index()
    if lc.AS_OF is not None:
        s = s[s.index <= lc.AS_OF]
    return s.dropna()


# ----------------------------------------------------------------------
# episodes
# ----------------------------------------------------------------------

EPISODE_GAP_WEEKS = 3     # interruptions of at most three weeks do not end an episode
EPISODE_MIN_WEEKS = 2     # an episode holds the marker in at least two weeks


def episodes_from_flags(flags: pd.Series, min_weeks: int = EPISODE_MIN_WEEKS,
                        gap_weeks: int = EPISODE_GAP_WEEKS,
                        run_min: int = 1) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Episodes of a weekly marker. Runs of consecutive marker weeks shorter
    than run_min are discarded first (the persistence requirement of a
    sustained marker such as S4); runs separated by at most gap_weeks
    unflagged weeks are then joined into one episode; an episode must hold
    the marker in at least min_weeks weeks. Returns (start, end, marker
    weeks) per episode."""
    runs: List[list] = []
    run = None
    for d, f in flags.sort_index().items():
        d = pd.Timestamp(d)
        if f:
            if run is not None and (d - run[1]).days <= 7:
                run[1], run[2] = d, run[2] + 1
            else:
                if run is not None:
                    runs.append(run)
                run = [d, d, 1]
        else:
            if run is not None:
                runs.append(run)
            run = None
    if run is not None:
        runs.append(run)
    runs = [r for r in runs if r[2] >= run_min]
    merged: List[list] = []
    for r in runs:
        if merged and (r[0] - merged[-1][1]).days <= 7 * (gap_weeks + 1):
            merged[-1][1] = r[1]
            merged[-1][2] += r[2]
        else:
            merged.append(list(r))
    return [tuple(m) for m in merged if m[2] >= min_weeks]


@dataclass
class MarkerResult:
    scenario: str
    flags: pd.Series                       # weekly boolean over the observable window
    episodes: List[Tuple[pd.Timestamp, pd.Timestamp, int]]
    window: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]
    rule: str
    thresholds: Dict[str, float] = field(default_factory=dict)

    @property
    def n_s(self) -> int:
        return len(self.episodes)

    @property
    def y_s(self) -> float:
        a, b = self.window
        if a is None or b is None:
            return 0.0
        return ((b - a).days + 7) / 365.25

    @property
    def episode_weeks(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.flags[self.flags].index)


def _align(series: Dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame({k: v for k, v in series.items() if len(v)}).sort_index()


def _exceed(s: pd.Series, q: float = MARKER_Q) -> Tuple[pd.Series, float]:
    thr = float(s.quantile(q))
    return s > thr, thr


def _window(df_or_s) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    idx = df_or_s.dropna(how="all").index if isinstance(df_or_s, pd.DataFrame) else df_or_s.dropna().index
    if len(idx) == 0:
        return None, None
    return pd.Timestamp(idx.min()), pd.Timestamp(idx.max())


def marker_S1() -> Optional[MarkerResult]:
    """Two or more of influenza, COVID 19 and RSV simultaneously above the P90
    of NHSN admissions or of NSSP ED share. The marker is observable in the
    weeks where all three pathogens are observed in at least one of the two
    series; the thresholds are computed over that observable window."""
    adm, ed = nhsn_admissions(), nssp_ed()
    paths = ["COVID-19", "Influenza", "RSV"]
    obs = {}
    for p in paths:
        parts = [x for x in (adm.get(p), ed.get(p)) if x is not None and len(x)]
        if not parts:
            return None
        obs[p] = pd.concat(parts, axis=1, sort=True).notna().any(axis=1)
    o = pd.DataFrame(obs).fillna(False)
    full = o.all(axis=1)
    if not full.any():
        return None
    w0, w1 = o.index[full].min(), o.index[full].max()
    inwin = lambda x: x[(x.index >= w0) & (x.index <= w1)]
    thr, per = {}, {}
    for p in paths:
        cols = []
        for lab, src in (("NHSN", adm), ("NSSP", ed)):
            if p in src and len(inwin(src[p])):
                f, t = _exceed(inwin(src[p]))
                thr[f"{lab} {p}"] = t
                cols.append(f)
        per[p] = pd.concat(cols, axis=1, sort=True).fillna(False).any(axis=1)
    df = pd.DataFrame(per).fillna(False)
    df = df[(df.index >= w0) & (df.index <= w1)]
    flags = df.sum(axis=1) >= 2
    return MarkerResult("S1", flags, episodes_from_flags(flags), (pd.Timestamp(w0), pd.Timestamp(w1)),
                        "two or more pathogens above P90 of NHSN admissions or NSSP ED share", thr)


def marker_S2() -> Optional[MarkerResult]:
    """Combined weekly respiratory admissions (NHSN, sum of the pathogens
    reported that week) above its P90."""
    df = _align(nhsn_admissions())
    if df.empty:
        return None
    comb = df.sum(axis=1, min_count=1).dropna()
    flags, t = _exceed(comb)
    return MarkerResult("S2", flags, episodes_from_flags(flags), _window(comb),
                        "combined NHSN admissions above P90", {"combined admissions": t})


SUMMER_MONTHS = (6, 7, 8, 9, 10)


def marker_S3() -> Optional[MarkerResult]:
    """Summer or early autumn (June to October) COVID 19 exceedance of the P90
    in NREVSS positivity, NSSP ED share, or NWSS wastewater."""
    parts, thr = {}, {}
    pos, ed, ww = nrevss_positivity(), nssp_ed(), wastewater_weekly()
    if "COVID-19" in pos:
        parts["pos"], thr["NREVSS COVID-19"] = _exceed(pos["COVID-19"])
    if "COVID-19" in ed:
        parts["ed"], thr["NSSP COVID-19"] = _exceed(ed["COVID-19"])
    if len(ww):
        parts["ww"], thr["NWSS Michigan"] = _exceed(ww)
    if not parts:
        return None
    df = pd.DataFrame({k: v.astype(float) for k, v in parts.items()}).sort_index()
    any_flag = df.fillna(0).max(axis=1) > 0
    summer = pd.Series(df.index.month.isin(SUMMER_MONTHS), index=df.index)
    flags = any_flag & summer
    return MarkerResult("S3", flags, episodes_from_flags(flags), _window(df),
                        "COVID 19 above P90 in positivity, ED share or wastewater, June to October", thr)


def marker_S4() -> Optional[MarkerResult]:
    """Weekly median of the CFA daily Rt above 1.0 for at least three
    consecutive weeks, any target pathogen."""
    rt = _align(rt_weekly())
    if rt.empty:
        return None
    union = pd.Series(False, index=rt.index)
    for c in rt.columns:
        f = (rt[c] > 1.0) & rt[c].notna()
        for a, b, n in episodes_from_flags(f, min_weeks=3, gap_weeks=0, run_min=3):
            union[(union.index >= a) & (union.index <= b)] = True
    return MarkerResult("S4", union, episodes_from_flags(union, min_weeks=3, gap_weeks=0), _window(rt),
                        "weekly median Rt above 1 for three or more consecutive weeks, any pathogen",
                        {"Rt": 1.0})


def marker_S5() -> Optional[MarkerResult]:
    """RSV above the P90 of NSSP ED share or NREVSS positivity."""
    parts, thr = {}, {}
    ed, pos = nssp_ed(), nrevss_positivity()
    if "RSV" in ed:
        parts["ed"], thr["NSSP RSV"] = _exceed(ed["RSV"])
    if "RSV" in pos:
        parts["pos"], thr["NREVSS RSV"] = _exceed(pos["RSV"])
    if not parts:
        return None
    df = pd.DataFrame({k: v.astype(float) for k, v in parts.items()}).sort_index()
    flags = df.fillna(0).max(axis=1) > 0
    return MarkerResult("S5", flags, episodes_from_flags(flags), _window(df),
                        "RSV above P90 of NSSP ED share or NREVSS positivity", thr)


def marker_S6() -> Optional[MarkerResult]:
    """RESP-NET combined weekly hospitalization rate, adults 65 and older,
    above its P90."""
    s = respnet_65()
    if s.empty:
        return None
    flags, t = _exceed(s)
    return MarkerResult("S6", flags, episodes_from_flags(flags), _window(s),
                        "RESP-NET combined 65+ weekly rate above P90", {"RESP-NET 65+": t})


def marker_S7() -> Optional[MarkerResult]:
    """NNDSS weekly confirmed and probable meningococcal cases above P90."""
    s = nndss_mening()
    if s.empty:
        return None
    flags, t = _exceed(s)
    return MarkerResult("S7", flags, episodes_from_flags(flags), _window(s),
                        "NNDSS weekly meningococcal cases above P90", {"NNDSS cases": t})


MARKERS: Dict[str, Callable[[], Optional[MarkerResult]]] = {
    "S1": marker_S1, "S2": marker_S2, "S3": marker_S3, "S4": marker_S4,
    "S5": marker_S5, "S6": marker_S6, "S7": marker_S7,
}


def compute_markers() -> Dict[str, MarkerResult]:
    out = {}
    for sid, fn in MARKERS.items():
        try:
            r = fn()
        except Exception:
            r = None
        if r is not None:
            out[sid] = r
    return out


# ----------------------------------------------------------------------
# marker magnitude and emphasis multipliers
# ----------------------------------------------------------------------

def _nssp_col(col: str) -> pd.Series:
    raw = lc._truncate(lc._nssp_jur(NAT), "week_end")
    if raw.empty or col not in raw.columns:
        return pd.Series(dtype=float)
    return pd.Series(pd.to_numeric(raw[col], errors="coerce").values,
                     index=pd.to_datetime(raw["week_end"]).values).sort_index().dropna()


def marker_magnitude(sid: str) -> pd.Series:
    """The ratio scale series measuring the intensity of each marker."""
    if sid == "S1":
        return _nssp_col("percent_visits_combined")
    if sid == "S2":
        df = _align(nhsn_admissions())
        return df.sum(axis=1, min_count=1).dropna() if not df.empty else pd.Series(dtype=float)
    if sid == "S3":
        return nssp_ed().get("COVID-19", pd.Series(dtype=float))
    if sid == "S4":
        rt = _align(rt_weekly())
        return rt.max(axis=1).dropna() if not rt.empty else pd.Series(dtype=float)
    if sid == "S5":
        return nssp_ed().get("RSV", pd.Series(dtype=float))
    if sid == "S6":
        return respnet_65()
    if sid == "S7":
        return nndss_mening()
    return pd.Series(dtype=float)


MAGNITUDE_LABEL = {
    "S1": "NSSP combined respiratory ED share", "S2": "NHSN combined admissions per 100k",
    "S3": "NSSP COVID 19 ED share", "S4": "weekly Rt, maximum over pathogens",
    "S5": "NSSP RSV ED share", "S6": "RESP-NET combined 65+ weekly rate",
    "S7": "NNDSS weekly meningococcal cases",
}


def episode_mask(index: pd.DatetimeIndex, m: MarkerResult) -> np.ndarray:
    mask = np.zeros(len(index), dtype=bool)
    for a, b, _ in m.episodes:
        mask |= (index >= a) & (index <= b)
    return mask


def elevation_ratio(series: pd.Series, m: MarkerResult) -> Tuple[float, int, int]:
    """Median of the series over the scenario's episode periods divided by its
    median over the scenario's observable window; nan when the series is
    not on a positive ratio scale or does not overlap the episodes."""
    s = series.dropna()
    a, b = m.window
    if s.empty or a is None:
        return np.nan, 0, 0
    s.index = pd.to_datetime(s.index)
    s = s[(s.index >= a) & (s.index <= b)]
    if s.empty or (s < 0).any():
        return np.nan, 0, len(s)
    ep = s[episode_mask(s.index, m)]
    base = float(s.median())
    if ep.empty or base <= 0:
        return np.nan, len(ep), len(s)
    return float(ep.median()) / base, len(ep), len(s)
