"""
scenario_engine.py

The scenario layer (thesis Chapter 2, "The Scenario Layer"; paper Section
4.6 and 4.7).

Registry      the curated Scenario Registry sheet: marker conditions, dated
              episodes, sources, status (weighted or stress test).
Frequency     f_s = n_s / Y_s. For scenarios with a countable weekly marker
              (S1 to S7) n_s and Y_s are recomputed every week from the feeds
              (modules.scenario_admission); event catalog scenarios (S8, S9)
              keep their documented counts. The curated counts stay visible.
Weights       w^(s): an elicited scenario specific profile when one covers
              the criteria; otherwise the agent's baseline weights with every
              emphasized criterion multiplied by its data derived emphasis
              multiplier k_j^(s) = max(1, median during the scenario's
              episodes / median over its observable window), from the
              criterion's own ratio scale series where one exists and from
              the scenario's marker magnitude otherwise; scenarios without a
              weekly series keep the baseline weights until experts weight
              them.
Limits        L^(s): registry overrides, and tightening: each emphasized
              criterion with a history takes the tightening quantile (P90,
              the quantile of the admission markers) of its history inside
              the configured window; Rt takes its absolute marker threshold 1.
Probability   pi_s = f_s^alpha v_s^beta h_s^gamma, normalized over the
              weighted scenarios; v_s and h_s from the fuzzy chain over the
              expert respondents and the auxiliary assessor, blended by
              lambda; full precision throughout, rounded only for display.
Stability     displacement against the baseline, robust or sensitive,
              exposure and instability, over the weighted scenarios.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules import state
from modules import scenario_admission as sa
from modules import live_subcriteria as ls
from modules import live_connectors as lc
from modules.load_data import load_scenario_registry
from modules.weight_manager import (INTENSITY_SCALE, gmir, aggregate_trapezoids, get_weights,
                                    normalize)
from modules import genai

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_TERM = "Moderate"
EMPHASIS_FILE = Path(__file__).resolve().parent.parent / "data" / "scenario_emphasis.json"
RT_CRITERION = "Effective Reproduction Number (Rt)"
GROWTH_CRITERION = "Probability of Outbreak Growth"


# ----------------------------------------------------------------------
# registry and admission
# ----------------------------------------------------------------------

def get_registry() -> pd.DataFrame:
    reg = load_scenario_registry().copy()
    reg["weighted"] = reg["Status"].astype(str).str.startswith("weighted")
    return reg


def scenario_ids(include_stress: bool = True) -> List[str]:
    reg = get_registry()
    return reg["Scenario ID"].tolist() if include_stress else \
        reg.loc[reg["weighted"], "Scenario ID"].tolist()


@state.cached
def markers() -> Dict[str, "sa.MarkerResult"]:
    s = state.settings()
    sa.EPISODE_GAP_WEEKS = int(s["episode_gap_weeks"])
    sa.EPISODE_MIN_WEEKS = int(s["episode_min_weeks"])
    out = {}
    for sid, fn in sa.MARKERS.items():
        try:
            m = fn()
        except Exception:
            m = None
        if m is None:
            continue
        if sid != "S4":
            m.episodes = sa.episodes_from_flags(m.flags, min_weeks=int(s["episode_min_weeks"]),
                                                gap_weeks=int(s["episode_gap_weeks"]))
        out[sid] = m
    return out


@state.cached
def admission_table() -> pd.DataFrame:
    """n_s, Y_s, f_s per scenario with their source, the recomputed episodes
    and the curated values side by side."""
    reg = get_registry()
    mk = markers()
    use_computed = state.setting("frequency_source") == "computed"
    rows = []
    for _, r in reg.iterrows():
        sid = r["Scenario ID"]
        n_reg = pd.to_numeric(r["n_s"], errors="coerce")
        y_reg = pd.to_numeric(r["Y_s (years observable)"], errors="coerce")
        m = mk.get(sid)
        replay = lc.AS_OF is not None and lc.AS_OF < REGISTRY_DATE
        if m is not None and use_computed and m.y_s > 0:
            n, y, src = float(m.n_s), float(m.y_s), "computed from the weekly series"
            eps = "; ".join(f"{a:%Y-%m-%d} to {b:%Y-%m-%d} ({k} wk)" for a, b, k in m.episodes)
            win = f"{m.window[0]:%Y-%m-%d} to {m.window[1]:%Y-%m-%d}"
            rule = m.rule
        elif replay and sid in sa.MARKERS and use_computed:
            # the marker series did not exist yet at the replay date
            n, y, src = 0.0, 0.0, "not observable at the replay date"
            eps, win, rule = "", "", str(r.get("Marker Condition", ""))
        else:
            n, y = (float(n_reg) if n_reg == n_reg else np.nan), (float(y_reg) if y_reg == y_reg else np.nan)
            src = "registry (event catalog)" if sid not in sa.MARKERS else "registry"
            eps, win, rule = str(r.get("Episodes", "")), "", str(r.get("Marker Condition", ""))
            if replay and n == n and y == y:
                # only the documented episodes dated on or before the replay date,
                # over the part of the registry window already elapsed
                starts = _episode_starts(eps)
                n = float(sum(1 for d in starts if d <= lc.AS_OF)) if starts else n
                y = max(0.0, y - (REGISTRY_DATE - lc.AS_OF).days / 365.25)
                src += ", truncated at the replay date"
        f = n / y if (y == y and y and y > 0 and n == n) else np.nan
        lo, hi = poisson_rate_interval(n, y)
        rows.append({"Scenario ID": sid, "Name": r["Name"], "weighted": bool(r["weighted"]),
                     "n_s": n, "Y_s": y, "f_s": f, "f_s 95% interval": (lo, hi),
                     "Source": src, "Episodes": eps, "Observable window": win, "Marker rule": rule,
                     "Registry n_s": n_reg, "Registry Y_s": y_reg})
    return pd.DataFrame(rows)


REGISTRY_DATE = pd.Timestamp("2026-09-27")   # date of the curated registry counts


def _episode_starts(text: str) -> List[pd.Timestamp]:
    """Start dates of the documented episodes, one per semicolon separated
    entry ('2021-12..2022-02 (note)', '2020 (note)', '2024-02')."""
    out = []
    for part in str(text).split(";"):
        m = re.match(r"\s*(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", part)
        if m:
            out.append(pd.Timestamp(year=int(m.group(1)), month=int(m.group(2) or 1),
                                    day=int(m.group(3) or 1)))
    return out


def poisson_rate_interval(n: float, y: float) -> Tuple[float, float]:
    """Exact (Garwood) 95% interval of an episode rate n / y."""
    if not (n == n and y == y) or y <= 0:
        return (np.nan, np.nan)
    from scipy.stats import chi2
    lo = 0.0 if n == 0 else chi2.ppf(0.025, 2 * n) / 2.0
    hi = chi2.ppf(0.975, 2 * n + 2) / 2.0
    return (float(lo / y), float(hi / y))


# ----------------------------------------------------------------------
# emphasis
# ----------------------------------------------------------------------

def emphasis_config() -> Dict[str, Dict[str, List[str]]]:
    try:
        return json.loads(EMPHASIS_FILE.read_text())["scenarios"]
    except Exception:
        return {}


def emphasized(scenario_id: str, agent_key: str) -> List[str]:
    return list(emphasis_config().get(scenario_id, {}).get(agent_key, []))


def _criterion_history(criterion: str) -> Optional[pd.Series]:
    """National weekly history of a criterion on a positive ratio scale,
    pooled over the covered diseases; None when there is none."""
    cs = ls.criterion_series("National", 1)
    if criterion in cs and cs[criterion]:
        s = pd.concat(list(cs[criterion].values())).dropna()
        return s if len(s) and (s >= 0).all() else None
    if criterion in (RT_CRITERION, GROWTH_CRITERION):
        from modules import live_connectors as lc
        sub = lc._rt_jur("National")
        if sub.empty:
            return None
        col = "median" if criterion == RT_CRITERION else "p_growing"
        parts = []
        for d in sub["disease"].dropna().unique():
            daily = lc._latest_run_daily(sub[sub["disease"] == d])
            parts.append(sa._weekly(pd.Series(pd.to_numeric(daily[col], errors="coerce").values,
                                              index=pd.to_datetime(daily["date"]).values)))
        s = pd.concat(parts).dropna() if parts else pd.Series(dtype=float)
        return s if len(s) else None
    return None


@state.cached
def emphasis_multipliers(scenario_id: str) -> pd.DataFrame:
    """One row per emphasized criterion: k_j^(s) and its basis."""
    cfg = emphasis_config().get(scenario_id, {})
    mk = markers().get(scenario_id)
    rows = []
    marker_k = np.nan
    if mk is not None and mk.episodes:
        r, ne, nw = sa.elevation_ratio(sa.marker_magnitude(scenario_id), mk)
        marker_k = r
    for agent, crits in cfg.items():
        for c in crits:
            k, basis = np.nan, ""
            if state.setting("emphasis_source") == "registry_k3":
                k, basis = 3.0, "fixed k = 3 (first generation default, ablation only)"
            elif mk is None:
                k, basis = 1.0, ("no weekly marker series (event catalog or stress test): "
                                 "baseline weight kept until experts weight this scenario")
            elif not mk.episodes:
                k, basis = 1.0, "no episode in the observable window: baseline weight kept"
            else:
                h = _criterion_history(c)
                if h is not None:
                    r, ne, nw = sa.elevation_ratio(h, mk)
                    if r == r:
                        k, basis = r, (f"own history: median over {ne} episode values / "
                                       f"median over {nw} window values")
                if not (k == k):
                    k, basis = marker_k, (f"scenario marker ({sa.MAGNITUDE_LABEL.get(scenario_id, '')}): "
                                          "criterion has no ratio scale weekly history")
                if not (k == k):
                    k, basis = 1.0, "no usable series: baseline weight kept"
            raw = k
            k = max(1.0, float(k))
            rows.append({"Scenario": scenario_id, "Agent": agent, "Criterion": c,
                         "Elevation ratio": raw, "Multiplier k": k,
                         "Basis": basis + (" (ratio below one, floored at one)" if raw < 1 else "")})
    return pd.DataFrame(rows)


def scenario_weights(scenario_id: str, agent_key: str,
                     criteria: List[str]) -> Tuple[Dict[str, float], str]:
    """w^(s) over `criteria`: an elicited scenario specific profile when it
    covers them; otherwise the baseline weights times the emphasis
    multipliers (Eq. scenario emphasis), normalized."""
    w_el, prov_el = get_weights(f"{agent_key}::{scenario_id}", criteria)
    if prov_el != "default":
        return w_el, f"scenario specific ({prov_el})"
    base, bprov = get_weights(agent_key, criteria)
    em = emphasis_multipliers(scenario_id)
    k = {r["Criterion"]: r["Multiplier k"] for _, r in em[em["Agent"] == agent_key].iterrows()} \
        if not em.empty else {}
    w = {c: base[c] * k.get(c, 1.0) for c in criteria}
    tag = ("fixed k = 3 emphasis" if state.setting("emphasis_source") == "registry_k3"
           else "data derived emphasis")
    return normalize(w), f"{tag} on {bprov} baseline weights"


# ----------------------------------------------------------------------
# limit overrides (tightening)
# ----------------------------------------------------------------------

def scenario_limit_overrides(scenario_id: str, agent_key: str,
                             jurisdiction: str = "National") -> Dict[str, float]:
    from modules.agent_engine import AGENTS
    crits = AGENTS[agent_key]["criteria"]
    dirs = AGENTS[agent_key]["directions"]
    out: Dict[str, float] = {}
    reg = get_registry().set_index("Scenario ID")
    if "Limit Overrides" in reg.columns and scenario_id in reg.index:
        hv = ls.historical_values(jurisdiction, 1, state.setting("history_window"))
        for part in str(reg.at[scenario_id, "Limit Overrides"]).split(";"):
            if "=" not in part:
                continue
            name, q = [x.strip() for x in part.split("=", 1)]
            m = re.match(r"^P(\d{1,2})$", q.upper())
            if name in crits and m and name in hv and len(hv[name]):
                out[name] = float(hv[name].quantile(int(m.group(1)) / 100.0))
    if bool(state.setting("scenario_tightening")):
        q = float(state.setting("tightening_quantile"))
        hv = ls.historical_values(jurisdiction, 1, state.setting("history_window"))
        for c in emphasized(scenario_id, agent_key):
            if c not in crits or c in out:
                continue
            if c == RT_CRITERION:
                out[c] = 1.0
            elif c in hv and len(hv[c]) >= 8:
                qq = q if dirs.get(c, "positive") == "positive" else 1.0 - q
                out[c] = float(hv[c].quantile(qq))
    return out


# ----------------------------------------------------------------------
# severity and system impact
# ----------------------------------------------------------------------

def get_assessments(scenario_id: str) -> dict:
    return state.get("intensity", scenario_id, {}) or {}


def set_expert_terms(scenario_id: str, severity: str, impact: str,
                     respondent: str = "expert") -> None:
    a = get_assessments(scenario_id)
    resp = dict(a.get("expert_respondents", {}))
    resp[respondent] = {"severity": severity, "impact": impact}
    a["expert_respondents"] = resp
    state.put("intensity", scenario_id, a)


def remove_expert(scenario_id: str, respondent: str) -> None:
    a = get_assessments(scenario_id)
    resp = dict(a.get("expert_respondents", {}))
    resp.pop(respondent, None)
    a["expert_respondents"] = resp
    state.put("intensity", scenario_id, a)


def set_ai_assessment(scenario_id: str, severity: str, impact: str,
                      justification: str, model: str) -> None:
    a = get_assessments(scenario_id)
    a["ai"] = {"severity": severity, "impact": impact, "justification": justification,
               "model": model, "time": state._now()}
    state.put("intensity", scenario_id, a)


def gemini_assess(scenario_row: pd.Series, model: str = DEFAULT_GEMINI_MODEL) -> Tuple[str, str, str]:
    terms = list(INTENSITY_SCALE)
    prompt = (
        "You are a public health decision support analyst. Rate the following "
        "historically observed health system scenario on two dimensions, choosing "
        f"EXACTLY one term for each from this scale: {terms}.\n"
        "severity: how severe outcomes are for affected people when the condition arises.\n"
        "system_impact: how far the condition strains health system capacity when it arises.\n\n"
        f"Scenario: {scenario_row['Name']}\nMarker condition: {scenario_row['Marker Condition']}\n"
        f"Documented episodes: {scenario_row['Episodes']}\nSources: {scenario_row['Sources']}\n\n"
        "Respond ONLY with a JSON object, no markdown fences, exactly: "
        '{"severity": "<term>", "system_impact": "<term>", "justification": "<two sentences>"}')
    key = os.environ.get("GEMINI_API_KEY", "")
    raw = genai._with_fallback(genai.call_gemini, key, model, prompt, temperature=0.0)
    txt = re.sub(r"```(json)?", "", raw).strip()
    data = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
    sev = str(data.get("severity", "")).title()
    imp = str(data.get("system_impact", "")).title()
    if sev not in INTENSITY_SCALE or imp not in INTENSITY_SCALE:
        raise ValueError(f"terms outside the scale: {sev}, {imp}")
    return sev, imp, str(data.get("justification", ""))


def _expert_value(a: dict, dim: str, rule: str) -> Tuple[Optional[float], int]:
    resp = a.get("expert_respondents", {})
    terms = [r[dim] for r in resp.values() if r.get(dim) in INTENSITY_SCALE]
    if not terms and a.get("expert", {}).get(dim) in INTENSITY_SCALE:   # earlier format
        terms = [a["expert"][dim]]
    if not terms:
        return None, 0
    return gmir(*aggregate_trapezoids([INTENSITY_SCALE[t] for t in terms], rule)), len(terms)


def _blend(scenario_id: str, dim: str, lam: float) -> Tuple[float, str]:
    a = get_assessments(scenario_id)
    ev, n = _expert_value(a, dim, state.setting("aggregation_rule"))
    ai_t = a.get("ai", {}).get(dim)
    av = gmir(*INTENSITY_SCALE[ai_t]) if ai_t in INTENSITY_SCALE else None
    if ev is not None and av is not None:
        return lam * ev + (1 - lam) * av, f"experts ({n}) and AI"
    if ev is not None:
        return ev, f"experts ({n})"
    if av is not None:
        return av, "AI only"
    return gmir(*INTENSITY_SCALE[DEFAULT_TERM]), "default (Moderate)"


@state.cached
def probability_table(lam: float = 0.7, alpha: float = 1.0, beta: float = 1.0,
                      gamma: float = 1.0) -> pd.DataFrame:
    adm = admission_table().set_index("Scenario ID")
    rows = []
    for sid, r in adm.iterrows():
        s_val, s_src = _blend(sid, "severity", lam)
        h_val, h_src = _blend(sid, "impact", lam)
        f = r["f_s"]
        pi = (f ** alpha) * (s_val ** beta) * (h_val ** gamma) \
            if r["weighted"] and f == f and f > 0 else np.nan
        rows.append({"Scenario ID": sid, "Name": r["Name"], "weighted": r["weighted"],
                     "n_s": r["n_s"], "Y_s": r["Y_s"], "f_s": f, "frequency source": r["Source"],
                     "v_s": s_val, "v source": s_src, "h_s": h_val, "h source": h_src, "pi_s": pi})
    df = pd.DataFrame(rows)
    total = df["pi_s"].sum(skipna=True)
    df["p_s"] = df["pi_s"] / total if total > 0 else np.nan
    return df


def probability_map(lam=0.7, alpha=1.0, beta=1.0, gamma=1.0) -> Dict[str, float]:
    df = probability_table(lam, alpha, beta, gamma)
    return {r["Scenario ID"]: float(r["p_s"]) for _, r in df.iterrows()
            if r["weighted"] and r["p_s"] == r["p_s"]}


# ----------------------------------------------------------------------
# re runs and stability
# ----------------------------------------------------------------------

def run_scenario(agent_key: str, scenario_id: Optional[str], jurisdiction: str = "National",
                 selected: Optional[Tuple[str, ...]] = None):
    from modules.agent_engine import stratify, environment, CORE4
    sel = tuple(selected) if selected else tuple(CORE4)
    w, wprov, _, _ = environment(agent_key, jurisdiction, sel, scenario_id)
    return stratify(agent_key, jurisdiction, sel, scenario_id), wprov


@state.cached
def stability_table(agent_key: str, jurisdiction: str = "National", lam=0.7, alpha=1.0,
                    beta=1.0, gamma=1.0, selected: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    base, _ = run_scenario(agent_key, None, jurisdiction, selected)
    p_map = probability_map(lam, alpha, beta, gamma)
    per = {sid: run_scenario(agent_key, sid, jurisdiction, selected)[0] for sid in p_map}
    rows = []
    for a in base.order:
        deltas = {sid: per[sid].tiers[a] - base.tiers[a] for sid in p_map if a in per[sid].tiers}
        trig = {sid: d for sid, d in deltas.items() if d != 0}
        rows.append({"Disease": a, "Baseline tier": base.tiers[a],
                     "Classification": "Robust priority" if not trig else "Scenario sensitive",
                     "Triggers (scenario: displacement)":
                         "; ".join(f"{s}: {'+' if d > 0 else ''}{d}" for s, d in trig.items()) or "none",
                     "Exposure E": sum(p_map[s] for s in trig),
                     "Instability I": sum(p_map[s] * abs(d) for s, d in trig.items())})
    for a in base.unstratified:
        rows.append({"Disease": a, "Baseline tier": None, "Classification": "Not stratifiable",
                     "Triggers (scenario: displacement)": "", "Exposure E": np.nan,
                     "Instability I": np.nan})
    return pd.DataFrame(rows)


@state.cached
def tier_matrix(agent_key: str, jurisdiction: str = "National",
                selected: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    base, _ = run_scenario(agent_key, None, jurisdiction, selected)
    cols = {"Baseline": base.tiers}
    for sid in scenario_ids(include_stress=True):
        res, _ = run_scenario(agent_key, sid, jurisdiction, selected)
        cols[sid if not sid.startswith("ST") else f"{sid} (stress test)"] = res.tiers
    return pd.DataFrame(cols).reindex(base.order)
