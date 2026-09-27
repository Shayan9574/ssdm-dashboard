"""
mosdm_core.py

Updated MOSDM for healthcare prioritization, implementing the agreed
specification (methods document, Section 9) exactly:

  9.2  orientation: every subcriterion carries a priority direction.
       "positive": larger value means greater need for prioritization.
       "negative": larger value means lower urgency.
  9.3  acceptable limits: median over the current pool by default,
       recomputed each iteration; expert overrides replace the median
       entirely and stay fixed across iterations.
  9.4  weighted attainment: k_i^w = sum_j w_j * a_ij, missing cells
       excluded, weights renormalized over available subcriteria.
  9.5  tier = { i : k_max^w - k_i^w < delta }, delta = min_j w_j over
       active subcriteria; dominance demotion guard (strictly worse on
       every shared subcriterion), configurable, default on.
  9.6  min max normalization in oriented space over the current pool;
       T0 = tier member with smallest weighted distance to the ideal
       point; D_ij = |xbar_ij - xbar_T0j|; S_i = sum_j w_j D_ij.
  9.7  iterate until every alternative holds a tier; nothing discarded.
  9.8  degenerate cases: singleton pool is its own tier; a subcriterion
       constant across the pool is flagged noninformative.

Scenario support (Sections 10 and 12): a scenario is a weight profile
plus optional acceptable limit overrides; pass them through `weights`
and `expert_limits`. The engine is pure Python over pandas and holds
no Streamlit dependency, so it is unit testable and cacheable.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

POSITIVE = "positive"   # larger value -> higher priority
NEGATIVE = "negative"   # larger value -> lower urgency


# ----------------------------------------------------------------------
# result containers
# ----------------------------------------------------------------------

@dataclass
class IterationRecord:
    iteration: int
    pool: List[str]
    limits: Dict[str, float]
    limit_source: Dict[str, str]          # "median" | "expert"
    attainment: Dict[str, Dict[str, float]]   # alt -> {criterion: 0/1/nan}
    k_weighted: Dict[str, float]
    delta: float
    candidate_tier: List[str]
    demotions: List[Tuple[str, str]]      # (demoted, dominated_by)
    tier: List[str]
    target: Optional[str]
    scores: Dict[str, float]              # S_i within the tier
    noninformative: List[str]
    missing: Dict[str, List[str]]         # alt -> criteria excluded
    thetas: Dict[str, float] = field(default_factory=dict)   # anchor per criterion
    attainment_mode: str = "binary"


@dataclass
class MOSDMResult:
    alternatives: List[str]
    criteria: List[str]
    directions: Dict[str, str]
    weights: Dict[str, float]
    tiers: Dict[str, int]                 # alt -> tier number (1 = top)
    order: List[str]                      # full ordering: tier, then S_i
    scores: Dict[str, float]              # S_i from the iteration in which
                                          # the alternative was tiered
    targets: Dict[int, str]               # tier -> target alternative
    separations: Dict[int, float]         # gap between tier t and t+1
    iterations: List[IterationRecord] = field(default_factory=list)
    explanations: Dict[str, str] = field(default_factory=dict)

    def table(self) -> pd.DataFrame:
        rows = []
        for rank, alt in enumerate(self.order, start=1):
            rows.append({
                "Overall Rank": rank,
                "Tier": self.tiers[alt],
                "Alternative": alt,
                "Weighted Distance S_i": round(self.scores[alt], 4),
                "Target of Tier": self.targets[self.tiers[alt]] == alt,
            })
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# core steps
# ----------------------------------------------------------------------

def _normalize_weights(weights: Dict[str, float], criteria: List[str]) -> Dict[str, float]:
    w = {c: float(weights.get(c, 0.0)) for c in criteria}
    total = sum(w.values())
    if total <= 0:
        n = len(criteria)
        return {c: 1.0 / n for c in criteria}
    return {c: v / total for c, v in w.items()}


def _median_limits(X: pd.DataFrame, pool: List[str], criteria: List[str]) -> Dict[str, float]:
    sub = X.loc[pool, criteria]
    return {c: float(sub[c].median(skipna=True)) for c in criteria}


def _attains(value: float, limit: float, direction: str) -> float:
    if pd.isna(value) or pd.isna(limit):
        return np.nan
    if direction == NEGATIVE:
        return 1.0 if value <= limit else 0.0
    return 1.0 if value >= limit else 0.0


def _oriented_minmax(X: pd.DataFrame, pool: List[str], criteria: List[str],
                     directions: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
    """Min max normalize over the pool so that 1 is the highest priority end.
    Returns the normalized frame and the list of noninformative criteria."""
    sub = X.loc[pool, criteria].astype(float)
    out = pd.DataFrame(index=sub.index, columns=criteria, dtype=float)
    flat = []
    for c in criteria:
        col = sub[c]
        cmin, cmax = col.min(skipna=True), col.max(skipna=True)
        if pd.isna(cmin) or pd.isna(cmax) or np.isclose(cmax, cmin):
            out[c] = 0.0 if col.notna().any() else np.nan
            out.loc[col.isna(), c] = np.nan
            flat.append(c)
            continue
        norm = (col - cmin) / (cmax - cmin)
        if directions.get(c, POSITIVE) == NEGATIVE:
            norm = 1.0 - norm
        out[c] = norm
    return out, flat


def _weighted_distance_to(row: pd.Series, ref: pd.Series,
                          weights: Dict[str, float]) -> float:
    avail = [c for c in weights if not pd.isna(row.get(c)) and not pd.isna(ref.get(c))]
    if not avail:
        return np.nan
    wsum = sum(weights[c] for c in avail)
    return float(sum(weights[c] * abs(row[c] - ref[c]) for c in avail) / wsum)


def _strictly_dominated(a: str, b: str, norm: pd.DataFrame,
                        criteria: List[str], min_shared: int = 1) -> bool:
    """True when alternative `a` is strictly worse than `b` (lower oriented
    value) on EVERY subcriterion for which both hold values. The guard
    abstains when fewer than `min_shared` criteria are shared, so a single
    surviving criterion under heavy missingness cannot decide dominance."""
    shared = [c for c in criteria
              if not pd.isna(norm.at[a, c]) and not pd.isna(norm.at[b, c])]
    if len(shared) < min_shared:
        return False
    return all(norm.at[a, c] < norm.at[b, c] for c in shared)




def _oriented_point(value: float, cmin: float, cmax: float, direction: str) -> float:
    if pd.isna(value) or pd.isna(cmin) or pd.isna(cmax) or np.isclose(cmax, cmin):
        return np.nan
    x = (value - cmin) / (cmax - cmin)
    x = min(max(x, 0.0), 1.0)
    return 1.0 - x if direction == NEGATIVE else x


def _graded_degree(xbar: float, lbar: float, theta: float) -> float:
    """Two regime value function anchored at the limit position lbar:
    0 at pool worst, theta exactly at the limit, 1 at pool best."""
    if pd.isna(xbar) or pd.isna(lbar):
        return np.nan
    if np.isclose(lbar, 0.0):
        return theta + (1.0 - theta) * xbar
    if np.isclose(lbar, 1.0):
        return theta * xbar
    if xbar <= lbar:
        return theta * (xbar / lbar)
    return theta + (1.0 - theta) * (xbar - lbar) / (1.0 - lbar)


def _calibrate_theta(xbars: "pd.Series", lbar: float,
                     bounds: Tuple[float, float]) -> float:
    """Discrimination calibrated anchor: the theta in bounds maximizing the
    variance of graded attainment over the pool; ties resolve toward 0.5."""
    vals = xbars.dropna().values
    if len(vals) < 2 or pd.isna(lbar):
        return 0.5
    grid = np.round(np.arange(bounds[0], bounds[1] + 1e-9, 0.01), 2)
    best_t, best_v = 0.5, -1.0
    for t in grid:
        g = np.array([_graded_degree(x, lbar, t) for x in vals])
        v = float(np.var(g))
        if v > best_v + 1e-12 or (abs(v - best_v) <= 1e-12 and
                                  abs(t - 0.5) < abs(best_t - 0.5)):
            best_v, best_t = v, t
    return float(best_t)

# ----------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------

def run_mosdm(
    matrix: pd.DataFrame,
    criteria: List[str],
    directions: Dict[str, str],
    weights: Optional[Dict[str, float]] = None,
    expert_limits: Optional[Dict[str, float]] = None,
    historical_limits: Optional[Dict[str, float]] = None,
    attainment_mode: str = "graded_calibrated",
    theta_fixed: float = 0.5,
    theta_bounds: Tuple[float, float] = (0.3, 0.7),
    dominance_guard: bool = True,
    guard_min_shared: int = 1,
    alternative_col: Optional[str] = None,
) -> MOSDMResult:
    """
    matrix         : DataFrame; rows are alternatives (index or a name
                     column via `alternative_col`), columns include the
                     criteria as numeric values (NaN = missing).
    criteria       : the subcriteria to evaluate, in display order.
    directions     : criterion -> "positive" | "negative".
    weights        : criterion -> weight; normalized internally; equal
                     weights when omitted (baseline default before the
                     Weight Manager supplies elicited values).
    expert_limits  : criterion -> threshold; replaces the median for that
                     criterion and stays fixed across iterations. Scenario
                     limit overrides L^(s) enter through this argument.
    dominance_guard: the tier demotion safeguard (default on).
    guard_min_shared: minimum number of shared criteria for a demotion
                     (default 1; raise it later if heavy missingness makes
                     single criterion dominance judgments too aggressive).
    historical_limits: criterion -> long run quantile threshold from the ten
                     year system history; second in the limits hierarchy
                     (expert > historical > pool median). Populated by the
                     scenario stage ETL; pool median applies when absent.
    attainment_mode: "graded_calibrated" (default; two regime value function
                     with the anchor theta_j chosen in theta_bounds to
                     maximize discrimination), "graded_fixed" (anchor =
                     theta_fixed), or "binary" (step function; the first
                     generation rule, kept for ablation).
    """
    X = matrix.copy()
    if alternative_col is not None:
        X = X.set_index(alternative_col)
    X = X[criteria].apply(pd.to_numeric, errors="coerce")

    alts = [str(a) for a in X.index]
    X.index = alts
    w = _normalize_weights(weights or {}, criteria)
    expert_limits = expert_limits or {}

    tiers: Dict[str, int] = {}
    scores: Dict[str, float] = {}
    targets: Dict[int, str] = {}
    iterations: List[IterationRecord] = []
    tier_last_scores: Dict[int, float] = {}
    tier_first_scores: Dict[int, float] = {}

    pool = list(alts)
    tier_no = 0

    while pool:
        tier_no += 1

        # -- 9.3 acceptable limits ------------------------------------
        med = _median_limits(X, pool, criteria)
        historical_limits = historical_limits or {}
        limits, limit_source = {}, {}
        for c in criteria:
            if c in expert_limits:
                limits[c], limit_source[c] = float(expert_limits[c]), "expert"
            elif c in historical_limits and not pd.isna(historical_limits[c]):
                limits[c], limit_source[c] = float(historical_limits[c]), "historical"
            else:
                limits[c], limit_source[c] = med[c], "median"

        # -- 9.4 weighted attainment ----------------------------------
        attain: Dict[str, Dict[str, float]] = {}
        k_w: Dict[str, float] = {}
        missing: Dict[str, List[str]] = {}
        thetas: Dict[str, float] = {}
        lbars: Dict[str, float] = {}
        stats = {c: (X.loc[pool, c].min(skipna=True),
                     X.loc[pool, c].max(skipna=True)) for c in criteria}
        for c in criteria:
            cmin, cmax = stats[c]
            lbars[c] = _oriented_point(limits[c], cmin, cmax,
                                       directions.get(c, POSITIVE))
            if attainment_mode == "graded_calibrated":
                xb = X.loc[pool, c].apply(
                    lambda v: _oriented_point(v, cmin, cmax,
                                              directions.get(c, POSITIVE)))
                thetas[c] = _calibrate_theta(xb, lbars[c], theta_bounds)
            elif attainment_mode == "graded_fixed":
                thetas[c] = float(theta_fixed)
        for a in pool:
            row = {}
            for c in criteria:
                if attainment_mode == "binary" or pd.isna(lbars[c]):
                    row[c] = _attains(X.at[a, c], limits[c],
                                      directions.get(c, POSITIVE))
                else:
                    cmin, cmax = stats[c]
                    xbar = _oriented_point(X.at[a, c], cmin, cmax,
                                           directions.get(c, POSITIVE))
                    row[c] = _graded_degree(xbar, lbars[c], thetas[c])
            attain[a] = row
            avail = [c for c in criteria if not pd.isna(row[c])]
            missing[a] = [c for c in criteria if pd.isna(row[c])]
            if avail:
                wsum = sum(w[c] for c in avail)
                k_w[a] = sum(w[c] * row[c] for c in avail) / wsum if wsum > 0 else 0.0
            else:
                k_w[a] = 0.0

        # -- 9.5 tier by tolerance ------------------------------------
        active = [c for c in criteria if X.loc[pool, c].notna().any()]
        delta = min((w[c] for c in active), default=1.0 / max(len(criteria), 1))
        k_max = max(k_w.values())
        candidate = [a for a in pool if (k_max - k_w[a]) < delta]

        # -- oriented normalization over the pool ---------------------
        norm, flat = _oriented_minmax(X, pool, criteria, directions)

        # -- dominance demotion guard ---------------------------------
        demotions: List[Tuple[str, str]] = []
        tier_members = list(candidate)
        if dominance_guard and len(candidate) > 1:
            for a in candidate:
                for b in candidate:
                    if a != b and _strictly_dominated(a, b, norm, criteria, guard_min_shared):
                        demotions.append((a, b))
                        if a in tier_members:
                            tier_members.remove(a)
                        break
            if not tier_members:          # never empty a tier entirely
                tier_members = list(candidate)
                demotions = []

        # -- 9.6 target reference and scores --------------------------
        ideal = pd.Series({c: (np.nan if norm[c].isna().all() else 1.0)
                           for c in criteria})
        dist_ideal = {a: _weighted_distance_to(norm.loc[a], ideal, w)
                      for a in tier_members}
        target = min(dist_ideal, key=lambda a: (dist_ideal[a], a))
        t_scores = {a: _weighted_distance_to(norm.loc[a], norm.loc[target], w)
                    for a in tier_members}
        ordered = sorted(tier_members, key=lambda a: (t_scores[a], a))

        # -- record ----------------------------------------------------
        for a in ordered:
            tiers[a] = tier_no
            scores[a] = t_scores[a]
        targets[tier_no] = target
        tier_first_scores[tier_no] = t_scores[ordered[0]]
        tier_last_scores[tier_no] = t_scores[ordered[-1]]

        iterations.append(IterationRecord(
            iteration=tier_no, pool=list(pool), limits=limits,
            limit_source=limit_source, attainment=attain, k_weighted=k_w,
            delta=delta, candidate_tier=candidate, demotions=demotions,
            tier=ordered, target=target, scores=t_scores,
            noninformative=flat, missing=missing,
            thetas=dict(thetas), attainment_mode=attainment_mode,
        ))

        pool = [a for a in pool if a not in tier_members]

    # separation: last score of tier t versus first score of tier t + 1
    separations = {t: round(tier_first_scores[t + 1] - tier_last_scores[t], 4)
                   for t in range(1, tier_no) if (t + 1) in tier_first_scores}

    order = sorted(alts, key=lambda a: (tiers[a], scores[a], a))

    explanations = {}
    for a in alts:
        rec = iterations[tiers[a] - 1]
        met = [c for c in criteria
               if not pd.isna(rec.attainment[a].get(c, np.nan))
               and rec.attainment[a][c] >= (rec.thetas.get(c, 1.0)
                                            if rec.attainment_mode != "binary" else 1.0)]
        expl = (f"Tier {tiers[a]}: weighted attainment "
                f"{rec.k_weighted[a]:.3f} (tolerance {rec.delta:.3f}, "
                f"mode {rec.attainment_mode}); reaches {len(met)} of "
                f"{len(criteria)} acceptable limits")
        if a == rec.target:
            expl += "; target state of its tier (closest to the ideal point)"
        demoted_by = [b for (d, b) in rec.demotions if d == a]
        for r in iterations[:tiers[a] - 1]:
            demoted_by += [b for (d, b) in r.demotions if d == a]
        if demoted_by:
            expl += f"; demoted by dominance guard (dominated by {demoted_by[0]})"
        explanations[a] = expl

    return MOSDMResult(
        alternatives=alts, criteria=criteria, directions=directions,
        weights=w, tiers=tiers, order=order, scores=scores,
        targets=targets, separations=separations,
        iterations=iterations, explanations=explanations,
    )
