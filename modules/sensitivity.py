"""
sensitivity.py

Sensitivity and simulation on the framework's parameters (checkpoint E
addition, approved in design review):

1. One at a time sweeps of lambda, alpha, beta, gamma: scenario
   probabilities and the final integrated ranks recomputed along a
   grid; rank trajectories per disease.
2. Monte Carlo robustness of the integration: agent importance weights
   W_g perturbed with a Dirichlet distribution centered on the active
   weights; the cross agent MOSDM reruns per draw (the second order
   matrix is fixed, so draws are cheap); outputs the probability of
   each tier per disease and the rank distribution.
3. Stability threshold sweep: quadrant membership as theta moves.
4. Attainment mode comparison at the integration level.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from modules import scenario_engine as se
from modules import ssdm_core as sc
from modules.agent_engine import AGENTS
from modules.mosdm_core import run_mosdm, NEGATIVE


def _final_ranks(jurisdiction, lam, a, b, g, mode) -> Dict[str, int]:
    _, res, _ = sc.cross_agent_run(jurisdiction, lam, a, b, g, mode, False)
    return {d: i + 1 for i, d in enumerate(res.order)}


@st.cache_data(ttl=21600, show_spinner=False)
def parameter_sweep(param: str, grid: Tuple[float, ...], jurisdiction: str,
                    lam: float, a: float, b: float, g: float,
                    mode: str) -> pd.DataFrame:
    rows = []
    for v in grid:
        args = {"lam": lam, "a": a, "b": b, "g": g}
        args[param] = v
        ranks = _final_ranks(jurisdiction, args["lam"], args["a"],
                             args["b"], args["g"], mode)
        for d, r in ranks.items():
            rows.append({"value": v, "Disease": d, "Final rank": r})
    return pd.DataFrame(rows)


@st.cache_data(ttl=21600, show_spinner=False)
def weight_monte_carlo(jurisdiction: str, lam: float, a: float, b: float,
                       g: float, mode: str, n_draws: int = 500,
                       concentration: float = 60.0, seed: int = 7
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perturbs W_g ~ Dirichlet(concentration * active weights); reruns the
    cross agent MOSDM per draw. Returns (tier probabilities, rank draws)."""
    M = sc.second_order(jurisdiction, lam, a, b, g, mode, False).dropna(how="all")
    crits = list(M.columns)
    w0, _ = sc.get_agent_weights()
    base = np.array([w0[k] for k in AGENTS], dtype=float)
    base = base / base.sum()
    rng = np.random.default_rng(seed)
    draws = rng.dirichlet(np.maximum(base * concentration, 1e-3), size=n_draws)
    Mr = M.reset_index().rename(columns={"index": "Disease Type"})
    tier_counts = {d: {} for d in M.index}
    rank_rows = []
    labels = [sc.AGENT_LABELS[k] for k in AGENTS]
    for w in draws:
        weights = dict(zip(labels, w))
        res = run_mosdm(Mr, crits, {c: NEGATIVE for c in crits},
                        weights=weights, attainment_mode=mode,
                        alternative_col="Disease Type")
        for i, d in enumerate(res.order):
            rank_rows.append({"Disease": d, "Rank": i + 1})
        for d, t in res.tiers.items():
            tier_counts[d][t] = tier_counts[d].get(t, 0) + 1
    tiers = pd.DataFrame([
        {"Disease": d,
         **{f"P(Tier {t})": round(c / n_draws, 3)
            for t, c in sorted(cnt.items())}}
        for d, cnt in tier_counts.items()]).fillna(0.0)
    return tiers, pd.DataFrame(rank_rows)


@st.cache_data(ttl=21600, show_spinner=False)
def prob_sweep(param: str, grid: Tuple[float, ...], lam: float, a: float,
               b: float, g: float) -> pd.DataFrame:
    """p_s for every weighted scenario along the parameter grid."""
    rows = []
    for v in grid:
        args = {"lam": lam, "a": a, "b": b, "g": g}
        args[param] = v
        pm = se.probability_map(args["lam"], args["a"], args["b"], args["g"])
        for sid, p in pm.items():
            rows.append({"value": v, "Scenario": sid, "p_s": p})
    return pd.DataFrame(rows)


@st.cache_data(ttl=21600, show_spinner=False)
def r_sweep(param: str, grid: Tuple[float, ...], agent_key: str,
            jurisdiction: str, lam: float, a: float, b: float, g: float,
            mode: str) -> pd.DataFrame:
    """Expected rank R per disease along the grid, for one agent: the
    continuous quantity underneath the discrete final ranks."""
    rows = []
    for v in grid:
        args = {"lam": lam, "a": a, "b": b, "g": g}
        args[param] = v
        ind = sc.indicators(agent_key, jurisdiction, args["lam"], args["a"],
                            args["b"], args["g"], mode)
        for _, r in ind.iterrows():
            rows.append({"value": v, "Disease": r["Disease"],
                         "R": r["R (expected rank)"]})
    return pd.DataFrame(rows)


@st.cache_data(ttl=21600, show_spinner=False)
def theta_sweep(agent_key: str, jurisdiction: str, lam: float, a: float,
                b: float, g: float, mode: str,
                grid: Tuple[float, ...] = tuple(np.round(
                    np.arange(0.5, 0.91, 0.05), 2))) -> pd.DataFrame:
    ind = sc.indicators(agent_key, jurisdiction, lam, a, b, g, mode)
    rows = []
    for th in grid:
        q = sc.quadrants(ind, th)
        for _, r in q.iterrows():
            rows.append({"theta": th, "Disease": r["Disease"],
                         "Quadrant": r["Quadrant"]})
    return pd.DataFrame(rows)


@st.cache_data(ttl=21600, show_spinner=False)
def mode_comparison(jurisdiction: str, lam: float, a: float, b: float,
                    g: float) -> pd.DataFrame:
    out = {}
    for mode in ("binary", "graded_fixed", "graded_calibrated"):
        _, res, _ = sc.cross_agent_run(jurisdiction, lam, a, b, g, mode, False)
        out[mode] = {d: f"Tier {res.tiers[d]} (rank {i+1})"
                     for i, d in enumerate(res.order)}
    return pd.DataFrame(out)
