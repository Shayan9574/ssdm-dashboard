"""
ssdm_core.py

Probability weighted indicators, the priority and stability typology, and
the integration of the eight perspectives (thesis Chapter 2, "Probability
Weighted Indicators", "The Priority and Stability Typology", "Integration
of Several Perspectives"; paper Sections 4.7 and 4.9).

  R_i     = sum_s p_s r_i^(s)                   expected rank
  Phi_i   = sum_s p_s 1[tier_i^(s) = 1]         dominance frequency
  sigma_i = sqrt(sum_s p_s (r_i^(s) - R_i)^2)   rank dispersion
  SI_i    = max(0, 1 - 2 sigma_i / (m - 1))     stability index
over the weighted scenarios only; the baseline is the reference. Ranks are
over the stratified alternatives; an alternative that cannot be stratified
under an agent is reported, and its cell of the second order matrix is
missing (excluded with weight renormalization), never a bottom rank.

Second order matrix: alternatives by perspectives, entries R_i^g (expected)
or the baseline rank; Agent 5 is reversed, R -> m + 1 - R. The core runs on
it with the perspective weights W_g, on the expected matrix and in
parallel on the baseline matrix. Every quantity is kept at full precision.
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from modules import state
from modules.agent_engine import AGENTS, CORE4
from modules.mosdm_core import run_mosdm, NEGATIVE
from modules import scenario_engine as se
from modules import weight_manager as wm
from modules import genai

INTEGRATION_INVERT = {"A5"}
AGENT_LABELS = {k: AGENTS[k]["title"].split(":")[0] for k in AGENTS}


def _rank_map(result) -> Dict[str, float]:
    """Overall ranks with mid ranks for exact ties (MOSDMResult.ranks)."""
    return result.ranks()


@state.cached
def indicators(agent_key: str, jurisdiction: str, lam: float, alpha: float,
               beta: float, gamma: float, mode: str = "",
               selected: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    p_map = se.probability_map(lam, alpha, beta, gamma)
    weighted = list(p_map)
    runs = {sid: se.run_scenario(agent_key, sid, jurisdiction, selected)[0] for sid in weighted}
    base, _ = se.run_scenario(agent_key, None, jurisdiction, selected)
    m = len(base.order)
    rows = []
    for d in base.order:
        ranks = {s: _rank_map(runs[s])[d] for s in weighted if d in runs[s].tiers}
        if len(ranks) < len(weighted):
            continue
        R = sum(p_map[s] * ranks[s] for s in weighted)
        Phi = sum(p_map[s] * (runs[s].tiers[d] == 1) for s in weighted)
        sig = float(np.sqrt(sum(p_map[s] * (ranks[s] - R) ** 2 for s in weighted)))
        SI = max(0.0, 1.0 - 2.0 * sig / (m - 1)) if m > 1 else 1.0
        rows.append({"Disease": d, "Baseline rank": _rank_map(base)[d],
                     "Baseline tier": base.tiers[d], "R (expected rank)": R,
                     "Phi (top tier prob.)": Phi, "sigma (dispersion)": sig,
                     "SI (stability)": SI})
    for d in base.unstratified:
        rows.append({"Disease": d, "Baseline rank": np.nan, "Baseline tier": np.nan,
                     "R (expected rank)": np.nan, "Phi (top tier prob.)": np.nan,
                     "sigma (dispersion)": np.nan, "SI (stability)": np.nan})
    return pd.DataFrame(rows)


TYPOLOGY = ["Structural priority", "Conditionally critical", "Stable low priority", "Latent risk"]


def quadrants(ind: pd.DataFrame, theta: float = 0.7) -> pd.DataFrame:
    out = ind.copy()
    m = int(out["R (expected rank)"].notna().sum())
    cut = (m + 1) / 2.0

    def cls(r):
        if pd.isna(r["R (expected rank)"]):
            return "Not stratifiable"
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


def get_agent_weights() -> Tuple[Dict[str, float], str]:
    return wm.get_weights("agent_level::global", list(AGENTS))


@state.cached
def second_order(jurisdiction: str, lam: float, alpha: float, beta: float, gamma: float,
                 mode: str = "", use_baseline: bool = False,
                 selected: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    alts = list(selected) if selected else list(CORE4)
    cols = {}
    for k in AGENTS:
        if use_baseline:
            base, _ = se.run_scenario(k, None, jurisdiction, selected)
            r = {d: float(v) for d, v in _rank_map(base).items()}
            m = len(base.order)
        else:
            ind = indicators(k, jurisdiction, lam, alpha, beta, gamma, mode, selected)
            r = {d: v for d, v in zip(ind["Disease"], ind["R (expected rank)"]) if v == v}
            m = int(ind["R (expected rank)"].notna().sum())
        if k in INTEGRATION_INVERT:
            r = {d: (m + 1 - v) for d, v in r.items()}
        cols[AGENT_LABELS[k]] = r
    return pd.DataFrame(cols).reindex(alts)


@state.cached
def cross_agent_run(jurisdiction: str, lam: float, alpha: float, beta: float, gamma: float,
                    mode: str = "", use_baseline: bool = False,
                    selected: Optional[Tuple[str, ...]] = None):
    M = second_order(jurisdiction, lam, alpha, beta, gamma, mode, use_baseline, selected)
    crits = list(M.columns)
    w, wprov = get_agent_weights()
    weights = {AGENT_LABELS[k]: w[k] for k in AGENTS}
    s = state.settings()
    res = run_mosdm(M.reset_index().rename(columns={"index": "Disease Type"}),
                    crits, {c: NEGATIVE for c in crits}, weights=weights,
                    attainment_mode=s["attainment_mode"], theta_fixed=float(s["theta_fixed"]),
                    theta_bounds=(float(s["theta_min"]), float(s["theta_max"])),
                    dominance_guard=bool(s["dominance_guard"]),
                    guard_min_shared=int(s["guard_min_shared"]),
                    alternative_col="Disease Type")
    return M, res, wprov


# ----------------------------------------------------------------------
# AI narrative (never alters a number)
# ----------------------------------------------------------------------

def _gemini(prompt: str, model: str) -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    return genai._with_fallback(genai.call_gemini, key, model, prompt, temperature=0.0)


def synthesize(final_table: pd.DataFrame, quad_tables: Dict[str, pd.DataFrame],
               prob: pd.DataFrame, external: Optional[pd.DataFrame] = None,
               model: str = se.DEFAULT_GEMINI_MODEL) -> str:
    prompt = (
        "You are writing the final synthesis of a public health prioritization run for "
        "policy makers. Using ONLY the numbers given, write five short paragraphs: the "
        "integrated prioritization and what distinguishes the tiers; which diseases are "
        "structural priorities and which are conditionally critical and under which "
        "scenarios; which scenarios carry the most weight and why; what the external "
        "evidence adds (it never changed the stratification); caveats (data windows, "
        "defaults in place). Do not invent numbers; quote only these.\n\n"
        f"Final integrated stratification:\n{final_table.to_string(index=False)}\n\n"
        + "\n".join(f"{k} typology:\n{q.to_string(index=False)}" for k, q in quad_tables.items())
        + f"\n\nScenario weights:\n{prob.to_string(index=False)}"
        + (f"\n\nExternal evidence (information only):\n{external.to_string(index=False)}"
           if external is not None and not external.empty else ""))
    return _gemini(prompt, model)


def interpret_tiers(agent_title: str, table: pd.DataFrame,
                    model: str = se.DEFAULT_GEMINI_MODEL) -> str:
    prompt = ("In three sentences, interpret this stratification for a public health "
              "audience. Use only the numbers shown; do not invent any.\n\n"
              f"{agent_title}\n{table.to_string(index=False)}")
    return _gemini(prompt, model)
