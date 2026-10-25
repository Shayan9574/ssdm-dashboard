"""
sensitivity.py

Sensitivity analyses available inside the dashboard (the full evaluation
protocol of the paper runs from scripts/experiments):

1. one at a time sweeps of lambda, alpha, beta, gamma through the full chain:
   scenario weights, expected ranks of one agent, final integrated ranks;
2. Monte Carlo acceptability of the integration under perturbed perspective
   weights W_g ~ Dirichlet(c * W_g): probability of each final tier and the
   rank distribution;
3. the stability threshold sweep of the typology;
4. the integrated result under each attainment mode.
Every computation reruns the engine with the current inputs and settings.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from modules import state
from modules import scenario_engine as se
from modules import ssdm_core as sc
from modules.agent_engine import AGENTS
from modules.mosdm_core import run_mosdm, NEGATIVE, MODES


def _final_ranks(jurisdiction, lam, a, b, g) -> Dict[str, float]:
    _, res, _ = sc.cross_agent_run(jurisdiction, lam, a, b, g)
    return res.ranks()


@state.cached
def parameter_sweep(param: str, grid: Tuple[float, ...], jurisdiction: str,
                    lam: float, a: float, b: float, g: float, mode: str = "") -> pd.DataFrame:
    rows = []
    for v in grid:
        args = {"lam": lam, "a": a, "b": b, "g": g, param: v}
        for d, r in _final_ranks(jurisdiction, args["lam"], args["a"], args["b"], args["g"]).items():
            rows.append({"value": v, "Disease": d, "Final rank": r})
    return pd.DataFrame(rows)


@state.cached
def weight_monte_carlo(jurisdiction: str, lam: float, a: float, b: float, g: float,
                       mode: str = "", n_draws: int = 500, concentration: float = 60.0,
                       seed: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    M = sc.second_order(jurisdiction, lam, a, b, g)
    crits = list(M.columns)
    w0, _ = sc.get_agent_weights()
    base = np.array([w0[k] for k in AGENTS], dtype=float)
    base = base / base.sum()
    rng = np.random.default_rng(seed)
    draws = rng.dirichlet(np.maximum(base * concentration, 1e-3), size=n_draws)
    Mr = M.reset_index().rename(columns={"index": "Disease Type"})
    s = state.settings()
    labels = [sc.AGENT_LABELS[k] for k in AGENTS]
    tier_counts: Dict[str, Dict[int, int]] = {d: {} for d in M.index}
    rank_rows = []
    for w in draws:
        res = run_mosdm(Mr, crits, {c: NEGATIVE for c in crits}, weights=dict(zip(labels, w)),
                        attainment_mode=s["attainment_mode"], theta_fixed=float(s["theta_fixed"]),
                        theta_bounds=(float(s["theta_min"]), float(s["theta_max"])),
                        dominance_guard=bool(s["dominance_guard"]),
                        guard_min_shared=int(s["guard_min_shared"]),
                        alternative_col="Disease Type")
        for d, r in res.ranks().items():
            rank_rows.append({"Disease": d, "Rank": r})
        for d, t in res.tiers.items():
            tier_counts[d][t] = tier_counts[d].get(t, 0) + 1
    tiers_all = sorted({t for c in tier_counts.values() for t in c})
    tiers = pd.DataFrame([{"Disease": d, **{f"P(Tier {t})": cnt.get(t, 0) / n_draws for t in tiers_all}}
                          for d, cnt in tier_counts.items()])
    return tiers, pd.DataFrame(rank_rows)


@state.cached
def prob_sweep(param: str, grid: Tuple[float, ...], lam: float, a: float, b: float,
               g: float) -> pd.DataFrame:
    rows = []
    for v in grid:
        args = {"lam": lam, "a": a, "b": b, "g": g, param: v}
        for sid, p in se.probability_map(args["lam"], args["a"], args["b"], args["g"]).items():
            rows.append({"value": v, "Scenario": sid, "p_s": p})
    return pd.DataFrame(rows)


@state.cached
def r_sweep(param: str, grid: Tuple[float, ...], agent_key: str, jurisdiction: str,
            lam: float, a: float, b: float, g: float, mode: str = "") -> pd.DataFrame:
    rows = []
    for v in grid:
        args = {"lam": lam, "a": a, "b": b, "g": g, param: v}
        ind = sc.indicators(agent_key, jurisdiction, args["lam"], args["a"], args["b"], args["g"])
        for _, r in ind.iterrows():
            rows.append({"value": v, "Disease": r["Disease"], "R": r["R (expected rank)"]})
    return pd.DataFrame(rows)


def theta_sweep(agent_key: str, jurisdiction: str, lam: float, a: float, b: float, g: float,
                mode: str = "", grid: Tuple[float, ...] = tuple(np.round(np.arange(0.5, 0.91, 0.05), 2))
                ) -> pd.DataFrame:
    ind = sc.indicators(agent_key, jurisdiction, lam, a, b, g)
    rows = []
    for th in grid:
        for _, r in sc.quadrants(ind, th).iterrows():
            rows.append({"theta": th, "Disease": r["Disease"], "Quadrant": r["Quadrant"]})
    return pd.DataFrame(rows)


def mode_comparison(jurisdiction: str, lam: float, a: float, b: float, g: float) -> pd.DataFrame:
    """The integrated result under each attainment mode; the session's mode is
    restored afterwards."""
    before = state._session().get("attainment_mode", None)
    out = {}
    try:
        for m in MODES:
            state.set_session_setting("attainment_mode", m)
            _, res, _ = sc.cross_agent_run(jurisdiction, lam, a, b, g)
            rk = res.ranks()
            out[m] = {d: f"Tier {res.tiers[d]} (rank {rk[d]:g})" for d in res.order}
    finally:
        if before is None:
            state._session().pop("attainment_mode", None)
        else:
            state.set_session_setting("attainment_mode", before)
    return pd.DataFrame(out)
