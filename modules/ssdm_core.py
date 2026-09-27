"""
ssdm_core.py

Checkpoint D: Sections 14 to 18 of the methods document.

Indicators (Section 14), per agent, over the weighted scenarios only
(the baseline stays outside the aggregation as the reference):
  R_i     = sum_s p_s r_i^(s)                expected rank
  Phi_i   = sum_s p_s 1[tier_i^(s) = 1]      top tier probability
  sigma_i = sqrt(sum_s p_s (r_i^(s) - R_i)^2)   rank dispersion
  SI_i    = max(0, 1 - 2 sigma_i / (n_A - 1))   stability index

Quadrants (Section 15): high priority when R_i <= (n_A + 1) / 2; high
stability when SI_i >= theta (default 0.7, configurable):
structural priority / conditionally critical / stable low priority /
latent risk.

Second order matrix (Section 16): rows diseases, columns agents,
entries R_i^g. Agent 5 operates in benefit space (rank 1 = strongest
prevention capacity), so its expected rank is inverted at integration
(r' = n_A + 1 - r) so that in every column a LOWER entry means greater
concern. Agent importance weights W_g come from the Weight Manager's
agent level scope (four input modes, equal weights default).

Cross agent SSDM (Section 17): the second generation MOSDM runs on the
second order matrix (direction negative: larger expected rank means
lower urgency), in parallel on the baseline rank matrix, giving the
final integrated tiers.

Synthesis (Section 18): Gemini writes the narrative from the computed
results; it NEVER alters any number.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from modules.agent_engine import AGENTS, CORE4
from modules.mosdm_core import run_mosdm, NEGATIVE
from modules import scenario_engine as se
from modules import weight_manager as wm
from modules import genai

INTEGRATION_INVERT = {"A5"}   # benefit space agents, inverted at integration
AGENT_LABELS = {k: AGENTS[k]["title"].split(":")[0] for k in AGENTS}


# ----------------------------------------------------------------------
# Section 14: indicators per agent
# ----------------------------------------------------------------------

def _rank_map(result) -> Dict[str, int]:
    return {a: i + 1 for i, a in enumerate(result.order)}


@st.cache_data(ttl=900, show_spinner=False)
def indicators(agent_key: str, jurisdiction: str, lam: float, alpha: float,
               beta: float, gamma: float, mode: str) -> pd.DataFrame:
    p_map = se.probability_map(lam, alpha, beta, gamma)
    weighted = [s for s in se.scenario_ids(include_stress=False) if s in p_map]
    runs = {sid: se.run_scenario(agent_key, sid, jurisdiction)[0]
            for sid in weighted}
    base, _ = se.run_scenario(agent_key, None, jurisdiction)
    diseases = base.order
    n_a = len(diseases)
    rows = []
    for d in diseases:
        ranks = {sid: _rank_map(runs[sid])[d] for sid in weighted}
        tops = {sid: 1.0 if runs[sid].tiers[d] == 1 else 0.0 for sid in weighted}
        R = sum(p_map[s] * ranks[s] for s in weighted)
        Phi = sum(p_map[s] * tops[s] for s in weighted)
        sig = float(np.sqrt(sum(p_map[s] * (ranks[s] - R) ** 2 for s in weighted)))
        SI = max(0.0, 1.0 - 2.0 * sig / (n_a - 1)) if n_a > 1 else 1.0
        rows.append({"Disease": d, "Baseline rank": _rank_map(base)[d],
                     "Baseline tier": base.tiers[d],
                     "R (expected rank)": round(R, 4),
                     "Phi (top tier prob.)": round(Phi, 4),
                     "sigma (dispersion)": round(sig, 4),
                     "SI (stability)": round(SI, 4)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Section 15: quadrant classification
# ----------------------------------------------------------------------

def quadrants(ind: pd.DataFrame, theta: float = 0.7) -> pd.DataFrame:
    n_a = len(ind)
    cut = (n_a + 1) / 2.0
    out = ind.copy()

    def cls(r):
        hi_p = r["R (expected rank)"] <= cut
        hi_s = r["SI (stability)"] >= theta
        if hi_p and hi_s:
            return "Structural priority"
        if hi_p:
            return "Conditionally critical"
        if hi_s:
            return "Stable low priority"
        return "Latent risk"

    out["Quadrant"] = out.apply(cls, axis=1)
    return out


# ----------------------------------------------------------------------
# Section 16: second order matrix and agent weights
# ----------------------------------------------------------------------

def get_agent_weights() -> Tuple[Dict[str, float], str]:
    return wm.get_weights("agent_level::global", list(AGENTS))


@st.cache_data(ttl=900, show_spinner=False)
def second_order(jurisdiction: str, lam: float, alpha: float, beta: float,
                 gamma: float, mode: str,
                 use_baseline: bool = False) -> pd.DataFrame:
    """Diseases by agents matrix of expected ranks (or baseline ranks when
    use_baseline); Agent 5 inverted so lower always means greater concern."""
    cols = {}
    for k in AGENTS:
        if use_baseline:
            base, _ = se.run_scenario(k, None, jurisdiction)
            r = {d: _rank_map(base).get(d, np.nan) for d in CORE4}
            n_a = len(base.order)
        else:
            ind = indicators(k, jurisdiction, lam, alpha, beta, gamma, mode)
            r = dict(zip(ind["Disease"], ind["R (expected rank)"]))
            n_a = len(ind)
        if k in INTEGRATION_INVERT:
            r = {d: (n_a + 1 - v) if v == v else v for d, v in r.items()}
        cols[AGENT_LABELS[k]] = r
    return pd.DataFrame(cols).reindex(CORE4)


# ----------------------------------------------------------------------
# Section 17: cross agent SSDM
# ----------------------------------------------------------------------

def cross_agent_run(jurisdiction: str, lam: float, alpha: float, beta: float,
                    gamma: float, mode: str, use_baseline: bool = False):
    M = second_order(jurisdiction, lam, alpha, beta, gamma, mode, use_baseline)
    M = M.dropna(how="all")
    crits = list(M.columns)
    w, wprov = get_agent_weights()
    weights = {AGENT_LABELS[k]: w[k] for k in AGENTS}
    res = run_mosdm(M.reset_index().rename(columns={"index": "Disease Type"}),
                    crits, {c: NEGATIVE for c in crits},
                    weights=weights, attainment_mode=mode,
                    alternative_col="Disease Type")
    return M, res, wprov


# ----------------------------------------------------------------------
# Section 18: AI synthesis and tier interpretation (never alters results)
# ----------------------------------------------------------------------

def _gemini(prompt: str, model: str) -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    return genai.call_gemini(key, model, prompt, temperature=0.3)


def synthesize(final_table: pd.DataFrame, quad_tables: Dict[str, pd.DataFrame],
               prob: pd.DataFrame, model: str = se.DEFAULT_GEMINI_MODEL) -> str:
    prompt = (
        "You are writing the final synthesis of a public health "
        "prioritization run for a methods dashboard. Using ONLY the numbers "
        "given, write four short paragraphs: the integrated prioritization "
        "and what distinguishes the tiers; which diseases are structurally "
        "prioritized versus conditionally critical and why (probability "
        "weighted stability); which scenarios carry the most probability "
        "mass and what that implies; one paragraph of caveats (data windows, "
        "defaults in place). Do not invent numbers; quote only these.\n\n"
        f"Final integrated stratification:\n{final_table.to_string(index=False)}\n\n"
        + "\n".join(f"Agent {k} quadrants:\n{q.to_string(index=False)}"
                    for k, q in quad_tables.items())
        + f"\n\nScenario probabilities:\n{prob.to_string(index=False)}")
    return _gemini(prompt, model)


def interpret_tiers(agent_title: str, table: pd.DataFrame,
                    model: str = se.DEFAULT_GEMINI_MODEL) -> str:
    prompt = (
        "In three sentences, interpret this stratification for a public "
        "health audience. Use only the numbers shown; do not invent any.\n\n"
        f"{agent_title}\n{table.to_string(index=False)}")
    return _gemini(prompt, model)
