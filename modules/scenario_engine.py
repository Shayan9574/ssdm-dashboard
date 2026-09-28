"""
scenario_engine.py

Checkpoint C2: Sections 10 to 13 of the methods document.

Registry: read from the 'Scenario Registry' curated sheet. A weighted
scenario carries a default weight profile derived from its emphasis
keywords (matched criteria weighted three to one, normalized), fully
replaceable per scenario through the Weight Manager's elicitation
modes. The ST stress test runs and displays like any scenario but is
excluded from probability estimation and every probability weighted
quantity.

Probability (Section 11): pi_s = f_s^alpha * s_s^beta * h_s^gamma with
f_s = n_s / Y_s from the registry, severity and system impact each
elicited on the four term fuzzy scale (GMIR) from experts and from the
Gemini assessor, blended with lambda (default 0.7, expert dominant),
normalized over the weighted scenarios only. The baseline stays outside
the aggregation as the stability reference.

Stability (Section 13): tier displacement against the baseline,
robust versus scenario sensitive, with triggers, exposure (summed p_s
over triggering scenarios) and the instability index sum p_s |delta|.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from modules.load_data import load_scenario_registry
from modules.weight_manager import INTENSITY_SCALE, gmir, get_entry, set_entry
from modules.mosdm_core import run_mosdm, MOSDMResult
from modules.agent_engine import AGENTS, CORE4
from modules.agent_m_data import get_active_decision_matrix
from modules import genai

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_TERM = "Moderate"

# emphasis keyword -> substrings matched against criterion names (lowercase)
EMPHASIS_ALIASES: Dict[str, List[str]] = {
    "hospitalization": ["hospitalization", "hosp rate"],
    "admissions": ["hospitalization", "admission"],
    "icu": ["icu"],
    "ed": ["ed visit"],
    "occupancy": ["occupancy"],
    "stay": ["length of stay"],
    "positivity": ["positivity"],
    "incidence": ["incidence"],
    "rt": ["reproduction number (rt)", "effective reproduction"],
    "r0": ["reproduction number (r0)", "basic reproduction"],
    "growth": ["growth"],
    "velocity": ["reproduction"],
    "wastewater": ["wastewater"],
    "pediatric": ["pediatric"],
    "geriatric": ["geriatric"],
    "fatality": ["fatality"],
    "cfr": ["fatality"],
    "complication": ["complication"],
    "onset": ["onset"],
    "window": ["treatment window"],
    "resistance": ["resistance"],
    "amr": ["resistance"],
    "treatment": ["treatment"],
    "vaccine": ["vaccine"],
    "effectiveness": ["effectiveness"],
    "coverage": ["coverage"],
    "npi": ["npi"],
    "seroprevalence": ["seroprevalence"],
    "rsv": [],   # disease selector keyword, not a criterion
    "disparity": ["disparity"],
    "cost": ["cost"],
    "economic": ["cost"],
    "svi": ["svi"],
}


# ----------------------------------------------------------------------
# registry and scenario weight profiles
# ----------------------------------------------------------------------

@st.cache_data(ttl=21600)
def get_registry() -> pd.DataFrame:
    reg = load_scenario_registry()
    reg["weighted"] = reg["Status"].astype(str).str.startswith("weighted")
    return reg


def scenario_ids(include_stress: bool = True) -> List[str]:
    reg = get_registry()
    ids = reg["Scenario ID"].tolist()
    if not include_stress:
        ids = reg.loc[reg["weighted"], "Scenario ID"].tolist()
    return ids


def emphasis_keywords(scenario_id: str) -> List[str]:
    reg = get_registry().set_index("Scenario ID")
    raw = str(reg.at[scenario_id, "Default Emphasis Keywords"])
    return [k.strip().lower() for k in raw.split(",") if k.strip()]


def scenario_weights(scenario_id: str, agent_key: str) -> Tuple[Dict[str, float], str]:
    """Weight profile w^(s) for one agent: elicited profile when stored,
    otherwise the emphasis derived default (matched criteria three to one)."""
    e = get_entry("criterion", f"{agent_key}::{scenario_id}")
    criteria = AGENTS[agent_key]["criteria"]
    if e and set(e["values"]) >= set(criteria):
        w = {c: e["values"][c] for c in criteria}
        t = sum(w.values())
        return {c: v / t for c, v in w.items()}, e["provenance"]
    kws = emphasis_keywords(scenario_id)
    subs = [s for k in kws for s in EMPHASIS_ALIASES.get(k, [k])]
    w = {}
    for c in criteria:
        cl = c.lower()
        w[c] = 3.0 if any(s in cl for s in subs) else 1.0
    t = sum(w.values())
    return {c: v / t for c, v in w.items()}, "emphasis default"


# ----------------------------------------------------------------------
# severity and system impact elicitation
# ----------------------------------------------------------------------

def _intensity_store() -> dict:
    if "scenario_intensity" not in st.session_state:
        st.session_state["scenario_intensity"] = {}
    return st.session_state["scenario_intensity"]


def set_expert_terms(scenario_id: str, severity: str, impact: str) -> None:
    _intensity_store()[scenario_id] = {
        **_intensity_store().get(scenario_id, {}),
        "expert": {"severity": severity, "impact": impact}}


def set_ai_assessment(scenario_id: str, severity: str, impact: str,
                      justification: str, model: str) -> None:
    _intensity_store()[scenario_id] = {
        **_intensity_store().get(scenario_id, {}),
        "ai": {"severity": severity, "impact": impact,
               "justification": justification, "model": model}}


def get_assessments(scenario_id: str) -> dict:
    return _intensity_store().get(scenario_id, {})


def gemini_assess(scenario_row: pd.Series,
                  model: str = DEFAULT_GEMINI_MODEL) -> Tuple[str, str, str]:
    """Asks Gemini for a severity and system impact rating with a written
    justification, grounded in the registry's documented episodes."""
    terms = list(INTENSITY_SCALE)
    prompt = (
        "You are a public health decision support analyst. Rate the "
        "following historically observed health system scenario on two "
        "dimensions, choosing EXACTLY one term for each from this scale: "
        f"{terms}.\n"
        "severity: how severe outcomes are for affected people when the "
        "condition arises.\n"
        "system_impact: how far the condition strains health system "
        "capacity when it arises.\n\n"
        f"Scenario: {scenario_row['Name']}\n"
        f"Marker condition: {scenario_row['Marker Condition']}\n"
        f"Documented episodes: {scenario_row['Episodes']}\n"
        f"Sources: {scenario_row['Sources']}\n\n"
        "Respond ONLY with a JSON object, no markdown fences, exactly: "
        '{"severity": "<term>", "system_impact": "<term>", '
        '"justification": "<two sentences>"}')
    key = os.environ.get("GEMINI_API_KEY", "")
    raw = genai._with_fallback(genai.call_gemini, key, model, prompt, temperature=0.2)
    txt = re.sub(r"```(json)?", "", raw).strip()
    data = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
    sev = str(data.get("severity", DEFAULT_TERM)).title()
    imp = str(data.get("system_impact", DEFAULT_TERM)).title()
    if sev not in INTENSITY_SCALE:
        sev = DEFAULT_TERM
    if imp not in INTENSITY_SCALE:
        imp = DEFAULT_TERM
    return sev, imp, str(data.get("justification", ""))


# ----------------------------------------------------------------------
# probability table (Section 11)
# ----------------------------------------------------------------------

def _blend(scenario_id: str, dim: str, lam: float) -> Tuple[float, str]:
    a = get_assessments(scenario_id)
    exp_t = a.get("expert", {}).get(dim)
    ai_t = a.get("ai", {}).get(dim)
    if exp_t and ai_t:
        return lam * gmir(*INTENSITY_SCALE[exp_t]) + (1 - lam) * gmir(*INTENSITY_SCALE[ai_t]), "expert+AI"
    if exp_t:
        return gmir(*INTENSITY_SCALE[exp_t]), "expert only"
    if ai_t:
        return gmir(*INTENSITY_SCALE[ai_t]), "AI only"
    return gmir(*INTENSITY_SCALE[DEFAULT_TERM]), "default (Moderate)"


def probability_table(lam: float = 0.7, alpha: float = 1.0,
                      beta: float = 1.0, gamma: float = 1.0) -> pd.DataFrame:
    reg = get_registry()
    rows = []
    for _, r in reg.iterrows():
        sid = r["Scenario ID"]
        n_s = float(pd.to_numeric(r["n_s"], errors="coerce") or 0)
        y_s = float(pd.to_numeric(r["Y_s (years observable)"], errors="coerce") or 0)
        f_s = n_s / y_s if y_s > 0 else np.nan
        a = get_assessments(sid)
        s_val, s_src = _blend(sid, "severity", lam)
        h_val, h_src = _blend(sid, "impact", lam)
        pi = (f_s ** alpha) * (s_val ** beta) * (h_val ** gamma) \
            if r["weighted"] and f_s and not np.isnan(f_s) else np.nan
        rows.append({
            "Scenario ID": sid, "Name": r["Name"], "Status": r["Status"],
            "n_s": n_s, "Y_s": y_s, "f_s": round(f_s, 4) if f_s == f_s else None,
            "Expert severity": a.get("expert", {}).get("severity"),
            "AI severity": a.get("ai", {}).get("severity"),
            "s_s": round(s_val, 4), "s source": s_src,
            "Expert impact": a.get("expert", {}).get("impact"),
            "AI impact": a.get("ai", {}).get("impact"),
            "h_s": round(h_val, 4), "h source": h_src,
            "pi_s": round(pi, 5) if pi == pi else None,
        })
    df = pd.DataFrame(rows)
    total = df.loc[df["pi_s"].notna(), "pi_s"].sum()
    df["p_s"] = df["pi_s"].apply(
        lambda v: round(v / total, 4) if v == v and v is not None and total > 0 else None)
    return df


def probability_map(lam=0.7, alpha=1.0, beta=1.0, gamma=1.0) -> Dict[str, float]:
    df = probability_table(lam, alpha, beta, gamma)
    return {r["Scenario ID"]: r["p_s"] for _, r in df.iterrows()
            if r["p_s"] is not None}


# ----------------------------------------------------------------------
# scenario re runs and stability (Sections 12 and 13)
# ----------------------------------------------------------------------

@st.cache_data(ttl=21600, show_spinner=False)
def _scenario_run(agent_key: str, scenario_id: Optional[str],
                  jurisdiction: str, mode: str,
                  weights_items: tuple, use_overlay: bool = False,
                  overlay_items: tuple = ()) -> MOSDMResult:
    cfg = AGENTS[agent_key]
    wide = get_active_decision_matrix(jurisdiction=jurisdiction)
    wide = wide[wide["Disease Type"].isin(CORE4)]
    profile = cfg["profile"](wide, jurisdiction)
    profile = profile[profile["Disease Type"].isin(CORE4)]
    if use_overlay and overlay_items:
        for (dis, crit), v in dict(overlay_items).items():
            m = profile["Disease Type"].astype(str) == dis
            if m.any() and crit in profile.columns:
                profile.loc[m, crit] = v
    return run_mosdm(profile, cfg["criteria"], cfg["directions"],
                     weights=dict(weights_items) or None,
                     attainment_mode=mode, alternative_col="Disease Type")


def run_scenario(agent_key: str, scenario_id: Optional[str],
                 jurisdiction: str = "National") -> Tuple[MOSDMResult, str]:
    """scenario_id None -> baseline run with the agent's elicited weights."""
    mode = st.session_state.get("attainment_mode", "graded_calibrated")
    if scenario_id is None:
        from modules import weight_manager as wm
        w, prov = wm.get_weights(agent_key, AGENTS[agent_key]["criteria"])
    else:
        w, prov = scenario_weights(scenario_id, agent_key)
    use_ov = bool(st.session_state.get("use_evidence_overlay", True))
    ov = ()
    if use_ov:
        from modules.evidence_gate import overlay_for
        ov = tuple(sorted(overlay_for(agent_key,
                                      AGENTS[agent_key]["criteria"]).items()))
    res = _scenario_run(agent_key, scenario_id, jurisdiction, mode,
                        tuple(sorted(w.items())), use_ov, ov)
    return res, prov


def stability_table(agent_key: str, jurisdiction: str = "National",
                    lam=0.7, alpha=1.0, beta=1.0, gamma=1.0) -> pd.DataFrame:
    """Section 13 for one agent: displacement per weighted scenario against
    the baseline; robust or scenario sensitive; exposure and instability."""
    base, _ = run_scenario(agent_key, None, jurisdiction)
    p_map = probability_map(lam, alpha, beta, gamma)
    weighted = scenario_ids(include_stress=False)
    rows = []
    per_scen = {sid: run_scenario(agent_key, sid, jurisdiction)[0]
                for sid in weighted}
    for a in base.order:
        deltas = {sid: per_scen[sid].tiers[a] - base.tiers[a]
                  for sid in weighted}
        triggers = {sid: d for sid, d in deltas.items() if d != 0}
        exposure = sum(p_map.get(sid, 0.0) for sid in triggers)
        instab = sum(p_map.get(sid, 0.0) * abs(d) for sid, d in triggers.items())
        rows.append({
            "Disease": a,
            "Baseline tier": base.tiers[a],
            "Classification": "Robust priority" if not triggers
                              else "Scenario sensitive",
            "Triggers (scenario: displacement)":
                "; ".join(f"{s}: {'+' if d>0 else ''}{d}"
                          for s, d in triggers.items()) or "none",
            "Exposure (sum p_s)": round(exposure, 4),
            "Instability index": round(instab, 4),
        })
    return pd.DataFrame(rows)


def tier_matrix(agent_key: str, jurisdiction: str = "National") -> pd.DataFrame:
    """Diseases by scenarios tier matrix for one agent, baseline first,
    stress test last with its label."""
    base, _ = run_scenario(agent_key, None, jurisdiction)
    cols = {"Baseline": base.tiers}
    for sid in scenario_ids(include_stress=True):
        res, _ = run_scenario(agent_key, sid, jurisdiction)
        label = sid if not sid.startswith("ST") else f"{sid} (stress test)"
        cols[label] = res.tiers
    return pd.DataFrame(cols).loc[base.order]
