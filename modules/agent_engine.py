"""
agent_engine.py

One place that (a) declares the eight agents (criteria, priority
directions, optimization space label, live chart hooks, profile builder)
and (b) runs the updated MOSDM engine for an agent with the active
weights and expert limits from the Weight Manager. All eight dashboard
pages render through agent_view.render_agent, so the engine runs once
per agent per configuration and is cached.

Priority directions follow each agent's own optimization space from the
methods document (Section 7 table of the About page): in every agent,
Tier 1 is that agent's "greatest concern / strongest signal" end; for
Agent 5 the space is benefit (Tier 1 = highest prevention capacity),
reconciled at the cross agent integration stage, not here.
"""

from typing import Callable, Dict, List
import pandas as pd
import streamlit as st

from modules.mosdm_core import run_mosdm, MOSDMResult, POSITIVE, NEGATIVE
from modules import weight_manager as wm
from modules.agent_m_data import get_active_decision_matrix
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


def _a7_profile(wide, jurisdiction):
    return agent7_outbreak.build_agent7_profile(wide, jurisdiction=jurisdiction)


def _a8_profile(wide, jurisdiction):
    return agent8_social.build_agent8_profile(wide)


AGENTS: Dict[str, dict] = {
    "A1": {
        "title": "Agent 1: Epidemiological Burden & Severity",
        "space": "Cost space: larger burden values mean higher priority.",
        "profile": lambda wide, jur: agent1_epidemiology.build_agent1_profile(wide),
        "criteria": _crit(agent1_epidemiology.SUBCRITERIA),
        "directions": _dir(agent1_epidemiology.SUBCRITERIA, {"cost": POSITIVE}),
        "charts": [
            ("Weekly Respiratory Admissions (per 100k)", get_hospital_admission_trends, True),
            ("Meningococcal Weekly Notifications (NNDSS)", get_meningitis_trends, True),
        ],
    },
    "A2": {
        "title": "Agent 2: Transmission & Susceptibility",
        "space": "Hybrid: seroprevalence protects (negative direction); all other criteria positive.",
        "profile": lambda wide, jur: agent2_transmission.build_agent2_profile(wide),
        "criteria": _crit(agent2_transmission.SUBCRITERIA),
        "directions": _dir(agent2_transmission.SUBCRITERIA,
                           {"cost": POSITIVE, "benefit": NEGATIVE}),
        "charts": [("Laboratory PCR Test Positivity (%)",
                    lambda jurisdiction: get_test_positivity_trends(), False)],
    },
    "A3": {
        "title": "Agent 3: Healthcare System Impact & Facility Strain",
        "space": "Cost space: longer stays and higher emergency volume mean higher priority.",
        "profile": lambda wide, jur: agent3_healthcare_impact.build_agent3_profile(wide),
        "criteria": _crit(agent3_healthcare_impact.SUBCRITERIA),
        "directions": _dir(agent3_healthcare_impact.SUBCRITERIA, {"cost": POSITIVE}),
        "charts": [
            ("Weekly ED Syndromic Visit Share (%)", get_ed_visit_trends, True),
            ("Hospital Capacity Occupancy (%)", get_hospital_occupancy_trends, True),
        ],
    },
    "A4": {
        "title": "Agent 4: Clinical Severity & Complexity",
        "space": "Inverted cost: shorter windows and faster deterioration mean higher urgency.",
        "profile": lambda wide, jur: agent4_clinical.build_agent4_profile(wide),
        "criteria": _crit(agent4_clinical.SUBCRITERIA),
        "directions": _dir(agent4_clinical.SUBCRITERIA,
                           {"cost": POSITIVE, "benefit_inverted": NEGATIVE}),
        "charts": [],
    },
    "A5": {
        "title": "Agent 5: Prevention & Control Feasibility",
        "space": "Benefit space: Tier 1 is the strongest prevention capacity (reconciled at cross agent integration).",
        "profile": lambda wide, jur: agent5_prevention.build_agent5_profile(wide),
        "criteria": _crit(agent5_prevention.SUBCRITERIA),
        "directions": _dir(agent5_prevention.SUBCRITERIA, {"benefit": POSITIVE}),
        "charts": [],
    },
    "A6": {
        "title": "Agent 6: Equity & Vulnerable Populations",
        "space": "Cost space: higher burden on children, seniors, and high disparity groups means higher priority.",
        "profile": lambda wide, jur: agent6_equity.build_agent6_profile(wide),
        "criteria": _crit(agent6_equity.SUBCRITERIA),
        "directions": _dir(agent6_equity.SUBCRITERIA, {"cost": POSITIVE}),
        "charts": [],
    },
    "A7": {
        "title": "Agent 7: Outbreak Dynamics & Surveillance",
        "space": "Cost space over live velocity: spread score, Rt, and growth probability, all positive.",
        "profile": _a7_profile,
        "criteria": ["Geographic Spread Risk (Score)",
                     "Effective Reproduction Number (Rt)",
                     "Probability of Outbreak Growth"],
        "directions": {"Geographic Spread Risk (Score)": POSITIVE,
                       "Effective Reproduction Number (Rt)": POSITIVE,
                       "Probability of Outbreak Growth": POSITIVE},
        "charts": [
            ("Real-Time Transmission Velocity (Rt)", get_rt_trends, True),
            ("SARS-CoV-2 Wastewater Viral Load (copies/L)", get_wastewater_trends, True),
        ],
    },
    "A8": {
        "title": "Agent 8: Social & Economic Impact",
        "space": "Cost space: higher annual direct medical cost means higher priority (single numeric criterion).",
        "profile": _a8_profile,
        "criteria": ["Estimated Annual Direct Medical Cost ($B)"],
        "directions": {"Estimated Annual Direct Medical Cost ($B)": POSITIVE},
        "charts": [],
    },
}


def median_limits_for(profile: pd.DataFrame, criteria: List[str]) -> Dict[str, float]:
    sub = profile.set_index("Disease Type")[criteria].apply(
        pd.to_numeric, errors="coerce")
    return {c: float(sub[c].median(skipna=True)) for c in criteria}


@st.cache_data(ttl=900, show_spinner=False)
def _cached_run(agent_key: str, jurisdiction: str,
                weights_items: tuple, limits_items: tuple,
                selected: tuple, attainment_mode: str) -> MOSDMResult:
    cfg = AGENTS[agent_key]
    wide = get_active_decision_matrix(jurisdiction=jurisdiction)
    wide = wide[wide["Disease Type"].isin(list(selected))]
    profile = cfg["profile"](wide, jurisdiction)
    profile = profile[profile["Disease Type"].isin(list(selected))]
    return run_mosdm(
        profile, cfg["criteria"], cfg["directions"],
        weights=dict(weights_items) or None,
        expert_limits=dict(limits_items) or None,
        attainment_mode=attainment_mode,
        alternative_col="Disease Type",
    )


def run_agent(agent_key: str, jurisdiction: str,
              selected: List[str]) -> tuple:
    """Returns (profile, result, weights, weight_provenance,
    expert_limits, limit_provenance) for an agent."""
    cfg = AGENTS[agent_key]
    wide = get_active_decision_matrix(jurisdiction=jurisdiction)
    wide = wide[wide["Disease Type"].isin(selected)]
    profile = cfg["profile"](wide, jurisdiction)
    profile = profile[profile["Disease Type"].isin(selected)]

    weights, wprov = wm.get_weights(agent_key, cfg["criteria"])
    limits, lprov = wm.get_expert_limits(agent_key)
    mode = st.session_state.get("attainment_mode", "graded_calibrated")
    result = _cached_run(agent_key, jurisdiction,
                         tuple(sorted(weights.items())),
                         tuple(sorted(limits.items())),
                         tuple(selected), mode)
    return profile, result, weights, wprov, limits, lprov
