"""
agent_engine.py

The eight domain agents (criteria, directions, profile builders, live
charts) and the one function that stratifies an agent's diseases under a
decision environment (baseline or a scenario), with every weight, limit
and method setting taken from modules.state.

In every agent Tier 1 is that agent's strongest claim to attention, except
Agent 5, which operates in benefit space (Tier 1 = strongest prevention
capacity) and is reversed at integration (Eq. ch2_reverse).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules import state
from modules.mosdm_core import run_mosdm, MOSDMResult, POSITIVE, NEGATIVE
from modules import weight_manager as wm
from modules.agent_m_data import get_active_decision_matrix
from modules import live_subcriteria as ls
from modules.utils_numeric import parse_fuzzy, parse_binary_fuzzy
from modules.live_connectors import (
    get_hospital_admission_trends, get_meningitis_trends,
    get_test_positivity_trends, get_ed_visit_trends,
    get_hospital_occupancy_trends, get_rt_trends, get_wastewater_trends,
)
from modules.agents import (
    agent1_epidemiology, agent2_transmission, agent3_healthcare_impact,
    agent4_clinical, agent5_prevention, agent6_equity, agent7_outbreak,
    agent8_social,
)

CORE4 = ["Influenza", "COVID-19", "Respiratory Syncytial Virus (RSV)",
         "Meningococcal Disease (Meningitis)"]


def _crit(subcriteria) -> List[str]:
    return [s["name"] for s in subcriteria]


def _dir(subcriteria, mapping: Dict[str, str]) -> Dict[str, str]:
    return {s["name"]: mapping.get(s["kind"], POSITIVE) for s in subcriteria}


CURATED_SUBCRITERIA = {
    "A1": agent1_epidemiology.SUBCRITERIA, "A2": agent2_transmission.SUBCRITERIA,
    "A3": agent3_healthcare_impact.SUBCRITERIA, "A4": agent4_clinical.SUBCRITERIA,
    "A5": agent5_prevention.SUBCRITERIA, "A6": agent6_equity.SUBCRITERIA,
}

AGENTS: Dict[str, dict] = {
    "A1": {
        "title": "Agent 1: Epidemiological Burden & Severity",
        "space": "Cost space: larger burden values mean higher priority.",
        "profile": lambda wide, jur: ls.attach("A1", agent1_epidemiology.build_agent1_profile(wide), jur),
        "criteria": _crit(agent1_epidemiology.SUBCRITERIA) + ls.criteria_for("A1"),
        "directions": _dir(agent1_epidemiology.SUBCRITERIA, {"cost": POSITIVE}),
        "charts": [("Weekly Respiratory Admissions (per 100k)", get_hospital_admission_trends),
                   ("Meningococcal Weekly Notifications (NNDSS)", get_meningitis_trends)],
    },
    "A2": {
        "title": "Agent 2: Transmission & Susceptibility",
        "space": "Hybrid: seroprevalence protects (negative direction); all other criteria positive.",
        "profile": lambda wide, jur: ls.attach("A2", agent2_transmission.build_agent2_profile(wide), jur),
        "criteria": _crit(agent2_transmission.SUBCRITERIA) + ls.criteria_for("A2"),
        "directions": _dir(agent2_transmission.SUBCRITERIA, {"cost": POSITIVE, "benefit": NEGATIVE}),
        "charts": [("Laboratory PCR Test Positivity (%), national", get_test_positivity_trends)],
    },
    "A3": {
        "title": "Agent 3: Healthcare System Impact & Facility Strain",
        "space": "Cost space: longer stays and higher emergency volume mean higher priority.",
        "profile": lambda wide, jur: ls.attach("A3", agent3_healthcare_impact.build_agent3_profile(wide), jur),
        "criteria": _crit(agent3_healthcare_impact.SUBCRITERIA) + ls.criteria_for("A3"),
        "directions": _dir(agent3_healthcare_impact.SUBCRITERIA, {"cost": POSITIVE}),
        "charts": [("Weekly ED Syndromic Visit Share (%)", get_ed_visit_trends),
                   ("Hospital Capacity Occupancy (%)", get_hospital_occupancy_trends)],
    },
    "A4": {
        "title": "Agent 4: Clinical Severity & Complexity",
        "space": "Inverted cost: shorter windows and faster deterioration mean higher urgency.",
        "profile": lambda wide, jur: agent4_clinical.build_agent4_profile(wide),
        "criteria": _crit(agent4_clinical.SUBCRITERIA),
        "directions": _dir(agent4_clinical.SUBCRITERIA, {"cost": POSITIVE, "benefit_inverted": NEGATIVE}),
        "charts": [],
    },
    "A5": {
        "title": "Agent 5: Prevention & Control Feasibility",
        "space": "Benefit space: Tier 1 is the strongest prevention capacity (reversed at integration).",
        "profile": lambda wide, jur: agent5_prevention.build_agent5_profile(wide),
        "criteria": _crit(agent5_prevention.SUBCRITERIA),
        "directions": _dir(agent5_prevention.SUBCRITERIA, {"benefit": POSITIVE}),
        "charts": [],
    },
    "A6": {
        "title": "Agent 6: Equity & Vulnerable Populations",
        "space": "Cost space: higher burden on children, seniors, and high disparity groups means higher priority.",
        "profile": lambda wide, jur: ls.attach("A6", agent6_equity.build_agent6_profile(wide), jur),
        "criteria": _crit(agent6_equity.SUBCRITERIA) + ls.criteria_for("A6"),
        "directions": _dir(agent6_equity.SUBCRITERIA, {"cost": POSITIVE}),
        "charts": [],
    },
    "A7": {
        "title": "Agent 7: Outbreak Dynamics & Surveillance",
        "space": "Cost space over live velocity: spread score, Rt, and growth probability, all positive.",
        "profile": lambda wide, jur: agent7_outbreak.build_agent7_profile(wide, jurisdiction=jur),
        "criteria": ["Geographic Spread Risk (Score)", "Effective Reproduction Number (Rt)",
                     "Probability of Outbreak Growth"],
        "directions": {"Geographic Spread Risk (Score)": POSITIVE,
                       "Effective Reproduction Number (Rt)": POSITIVE,
                       "Probability of Outbreak Growth": POSITIVE},
        "charts": [("Effective Reproduction Number (Rt), latest model run", get_rt_trends),
                   ("SARS-CoV-2 Wastewater, Michigan sewersheds (copies/L)", get_wastewater_trends)],
    },
    "A8": {
        "title": "Agent 8: Social & Economic Impact",
        "space": ("Cost space: higher annual direct medical cost means higher priority. The curated "
                  "SVI column holds a categorical driver, not a multiplier, so cost is the single "
                  "numeric criterion until SVI values are curated."),
        "profile": lambda wide, jur: agent8_social.build_agent8_profile(wide),
        "criteria": ["Estimated Annual Direct Medical Cost ($B)"],
        "directions": {"Estimated Annual Direct Medical Cost ($B)": POSITIVE},
        "charts": [],
    },
}
for _k in ("A1", "A2", "A3", "A6"):
    for _c in ls.criteria_for(_k):
        AGENTS[_k]["directions"][_c] = POSITIVE


def active_criteria(profile: pd.DataFrame, agent_key: str) -> List[str]:
    return ls.active_criteria(profile, AGENTS[agent_key]["criteria"])


# ----------------------------------------------------------------------
# profiles
# ----------------------------------------------------------------------

@state.cached
def profile(agent_key: str, jurisdiction: str, selected: Tuple[str, ...]) -> pd.DataFrame:
    wide = get_active_decision_matrix(jurisdiction=jurisdiction)
    wide = wide[wide["Disease Type"].isin(list(selected))]
    p = AGENTS[agent_key]["profile"](wide, jurisdiction)
    return p[p["Disease Type"].isin(list(selected))].reset_index(drop=True)


@state.cached
def fuzzy_profile(agent_key: str, jurisdiction: str, selected: Tuple[str, ...]) -> Dict[str, Dict[str, tuple]]:
    """alt -> criterion -> trapezoid for the curated criteria (ranges and
    bounds kept fuzzy); live criteria are crisp, degenerate trapezoids."""
    wide = get_active_decision_matrix(jurisdiction=jurisdiction)
    wide = wide[wide["Disease Type"].isin(list(selected))]
    prof = profile(agent_key, jurisdiction, selected).set_index("Disease Type")
    out: Dict[str, Dict[str, tuple]] = {}
    subs = {s["name"]: s for s in CURATED_SUBCRITERIA.get(agent_key, [])}
    for _, row in wide.iterrows():
        d = str(row["Disease Type"])
        out[d] = {}
        for c in AGENTS[agent_key]["criteria"]:
            if c in subs and subs[c]["id"] in wide.columns:
                raw = row[subs[c]["id"]]
                t = parse_binary_fuzzy(raw) if "_y_n" in subs[c]["id"] else parse_fuzzy(raw)
            elif d in prof.index and c in prof.columns and pd.notna(prof.at[d, c]):
                v = float(prof.at[d, c]); t = (v, v, v, v)
            else:
                t = None
            if t is not None:
                out[d][c] = t
    return out


def historical_limits_for(criteria: List[str], jurisdiction: str) -> Dict[str, float]:
    """Historical level of the hierarchy: the median of each live criterion's
    history inside the configured window."""
    stats = ls.historical_stats(jurisdiction, 1, state.setting("history_window"))
    return {c: stats[c]["p50"] for c in criteria if c in stats}


def default_limits(agent_key: str, jurisdiction: str,
                   selected: Tuple[str, ...]) -> Dict[str, Tuple[float, str]]:
    """The limit each criterion takes at the first iteration without an
    expert value, with its level of the hierarchy."""
    prof = profile(agent_key, jurisdiction, selected)
    hist = historical_limits_for(AGENTS[agent_key]["criteria"], jurisdiction)
    out = {}
    for c in AGENTS[agent_key]["criteria"]:
        if c in hist:
            out[c] = (hist[c], f"historical median ({state.setting('history_window')})")
        elif c in prof.columns:
            v = pd.to_numeric(prof[c], errors="coerce").dropna()
            out[c] = (float(v.median()) if len(v) >= 2 else np.nan,
                      "pool median" if len(v) >= 2 else "inactive (coverage)")
    return out


# ----------------------------------------------------------------------
# stratification
# ----------------------------------------------------------------------

def environment(agent_key: str, jurisdiction: str, selected: Tuple[str, ...],
                scenario_id: Optional[str] = None):
    """(weights, weight provenance, limits, limit provenance) of an agent
    under the baseline or a scenario."""
    prof = profile(agent_key, jurisdiction, selected)
    act = active_criteria(prof, agent_key)
    if scenario_id is None:
        w, wprov = wm.get_weights(agent_key, act)
        lim, lprov = wm.get_expert_limits(agent_key)
    else:
        from modules import scenario_engine as se
        w, wprov = se.scenario_weights(scenario_id, agent_key, act)
        lim, lprov = wm.get_expert_limits(agent_key)
        lim = dict(lim)
        ov = se.scenario_limit_overrides(scenario_id, agent_key, jurisdiction)
        dirs = AGENTS[agent_key]["directions"]
        for c, v in ov.items():
            # a scenario only tightens: where an expert limit exists, the
            # stricter of the two applies
            if c in lim:
                lim[c] = max(lim[c], v) if dirs.get(c, POSITIVE) == POSITIVE else min(lim[c], v)
            else:
                lim[c] = v
        if ov:
            lprov = f"{lprov}; scenario tightening on {len(ov)} criterion(a)"
    return w, wprov, lim, lprov


@state.cached
def stratify(agent_key: str, jurisdiction: str, selected: Tuple[str, ...],
             scenario_id: Optional[str] = None) -> MOSDMResult:
    cfg = AGENTS[agent_key]
    prof = profile(agent_key, jurisdiction, selected)
    act = active_criteria(prof, agent_key)
    w, _, lim, _ = environment(agent_key, jurisdiction, selected, scenario_id)
    s = state.settings()
    fz = fuzzy_profile(agent_key, jurisdiction, selected) if s["fuzzy_distance"] else None
    return run_mosdm(
        prof, act, cfg["directions"], weights=w, expert_limits=lim,
        historical_limits=historical_limits_for(act, jurisdiction),
        attainment_mode=s["attainment_mode"], theta_fixed=float(s["theta_fixed"]),
        theta_bounds=(float(s["theta_min"]), float(s["theta_max"])),
        dominance_guard=bool(s["dominance_guard"]), guard_min_shared=int(s["guard_min_shared"]),
        alternative_col="Disease Type", fuzzy=fz, fuzzy_distance=bool(s["fuzzy_distance"]))


def run_agent(agent_key: str, jurisdiction: str, selected: List[str],
              scenario_id: Optional[str] = None) -> tuple:
    """(profile, result, weights, weight provenance, limits, limit provenance)."""
    sel = tuple(selected)
    prof = profile(agent_key, jurisdiction, sel)
    w, wprov, lim, lprov = environment(agent_key, jurisdiction, sel, scenario_id)
    return prof, stratify(agent_key, jurisdiction, sel, scenario_id), w, wprov, lim, lprov
