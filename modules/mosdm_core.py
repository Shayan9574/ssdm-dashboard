"""
mosdm_core.py

The extended stratification core (thesis Chapter 2, Section "The Extended
Stratification Core"; paper Section 4). Pure Python over pandas, no
Streamlit dependency, so it is unit testable and cacheable.

Every iteration t of the core, over the current pool P_t:

  coverage   a criterion is active when it is observed for at least two
             alternatives of the initial pool; an alternative observed on
             no active criterion is reported as unstratified (never placed
             at the bottom, because an absence of evidence is not an
             absence of need).
  limits     expert > historical quantile > pool median (Eq. ch2_limits).
             The pool median is recomputed while at least two pool members
             are observed on the criterion; below that it stays frozen at
             its last value, so a lone observation is never judged against
             its own value.
  attainment graded two regime value function anchored at the limit, with
             the limit position measured over the pool together with the
             limit itself, so an alternative below an external limit never
             receives the credit of reaching it (Eq. ch2_attainment).
             Anchor modes: fixed, calibrated (variance maximizing, ties to
             the midpoint of the bounds), binary (first generation rule).
  weighted   K_i over the observed active criteria, weights renormalized.
  tier       K_max - K_i < delta, delta = smallest positive weight, strict
             with a 1e-9 numerical guard.
  guard      strict dominance demotion on oriented normalized values with
             at least q_min shared criteria; never empties a tier.
  target     tier member closest to the ideal point; within tier order by
             the weighted distance S_i to the target (crisp, or the
             discretized fuzzy distance of Eq. ch2_discretize).
  margin     mu_t = min K in the tier - max K left in the pool.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

POSITIVE = "positive"   # larger value -> stronger claim (higher tier)
NEGATIVE = "negative"   # larger value -> weaker claim

TOL_EPS = 1e-9          # strict tolerance guard (tier boundary)
TIE_EPS = 1e-12         # equality tolerance for tie breaking
MIN_COVERAGE = 2        # coverage rule: observed for at least two alternatives

# Defaults selected by the preregistered, outcome blind baseline selection
# (paper Section 7.1): graded attainment with fixed anchor 0.3, dominance
# guard on, scenario limit tightening on. Every alternative stays selectable.
DEFAULT_MODE = "graded_fixed"
DEFAULT_THETA = 0.3
DEFAULT_BOUNDS = (0.3, 0.7)
DEFAULT_GUARD = True
DEFAULT_QMIN = 1
DEFAULT_TIGHTENING = True

MODES = ("graded_fixed", "graded_calibrated", "binary")


# ----------------------------------------------------------------------
# result containers
# ----------------------------------------------------------------------

@dataclass
class IterationRecord:
    iteration: int
    pool: List[str]
    limits: Dict[str, float]
    limit_source: Dict[str, str]          # expert | historical | median | median (frozen)
    attainment: Dict[str, Dict[str, float]]   # alt -> {criterion: degree or nan}
    k_weighted: Dict[str, float]
    delta: float
    candidate_tier: List[str]
    demotions: List[Tuple[str, str]]      # (demoted, dominated_by)
    tier: List[str]
    target: Optional[str]
    scores: Dict[str, float]              # S_i within the tier
    noninformative: List[str]
    missing: Dict[str, List[str]]         # alt -> criteria excluded
    thetas: Dict[str, float] = field(default_factory=dict)
    attainment_mode: str = "binary"
    limit_positions: Dict[str, float] = field(default_factory=dict)
    ideal_distance: Dict[str, float] = field(default_factory=dict)
    reached: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class MOSDMResult:
    alternatives: List[str]               # stratified alternatives
    criteria: List[str]                   # active criteria
    directions: Dict[str, str]
    weights: Dict[str, float]             # normalized over active criteria
    tiers: Dict[str, int]                 # alt -> tier number (1 = top)
    order: List[str]                      # tier, then S_i
    scores: Dict[str, float]
    targets: Dict[int, str]
    separations: Dict[int, float]         # attainment margin mu_t
    iterations: List[IterationRecord] = field(default_factory=list)
    explanations: Dict[str, str] = field(default_factory=dict)
    unstratified: List[str] = field(default_factory=list)
    inactive_criteria: List[str] = field(default_factory=list)
    settings: Dict[str, object] = field(default_factory=dict)

    def ranks(self) -> Dict[str, float]:
        """Overall ranks ordered by tier, then by the distance S_i to the tier
        target. Alternatives of one tier with equal distances (within the
        tie tolerance, or both undefined) are tied and share the average of
        the positions they occupy (mid ranks), so no rank depends on the
        order of the names."""
        def skey(a):
            v = self.scores.get(a, np.nan)
            return (self.tiers[a], "nan" if pd.isna(v) else round(v / 1e-9))
        out, i = {}, 0
        while i < len(self.order):
            j = i
            while j + 1 < len(self.order) and skey(self.order[j + 1]) == skey(self.order[i]):
                j += 1
            mid = (i + 1 + j + 1) / 2.0
            for a in self.order[i:j + 1]:
                out[a] = mid
            i = j + 1
        return out

    def table(self) -> pd.DataFrame:
        rows = []
        rk = self.ranks()
        for alt in self.order:
            rank = rk[alt]
            rows.append({
                "Overall Rank": rank,
                "Tier": self.tiers[alt],
                "Alternative": alt,
                "Weighted Distance S_i": (round(self.scores[alt], 4)
                                          if self.scores[alt] == self.scores[alt] else None),
                "Target of Tier": self.targets[self.tiers[alt]] == alt,
            })
        for alt in self.unstratified:
            rows.append({"Overall Rank": None, "Tier": None, "Alternative": alt,
                         "Weighted Distance S_i": None, "Target of Tier": False})
        return pd.DataFrame(rows)

    @property
    def margins(self) -> Dict[int, float]:
        return self.separations


# ----------------------------------------------------------------------
# elementary functions
# ----------------------------------------------------------------------

def _normalize_weights(weights: Dict[str, float], criteria: List[str]) -> Dict[str, float]:
    w = {c: max(float(weights.get(c, 0.0)), 0.0) for c in criteria}
    total = sum(w.values())
    if total <= 0:
        n = max(len(criteria), 1)
        return {c: 1.0 / n for c in criteria}
    return {c: v / total for c, v in w.items()}


def _attains(value: float, limit: float, direction: str) -> float:
    if pd.isna(value) or pd.isna(limit):
        return np.nan
    if direction == NEGATIVE:
        return 1.0 if value <= limit else 0.0
    return 1.0 if value >= limit else 0.0


def _orient(value: float, lo: float, hi: float, direction: str) -> float:
    """Oriented min max position of a value in [lo, hi]; nan when the range
    is degenerate."""
    if pd.isna(value) or pd.isna(lo) or pd.isna(hi) or np.isclose(hi, lo):
        return np.nan
    x = (value - lo) / (hi - lo)
    x = min(max(x, 0.0), 1.0)
    return 1.0 - x if direction == NEGATIVE else x


def _graded_degree(xbar: float, lbar: float, theta: float) -> float:
    """Two regime value function: 0 at the weakest end, theta exactly at the
    limit, 1 at the strongest end (Eq. ch2_attainment with its boundary
    conventions)."""
    if pd.isna(xbar) or pd.isna(lbar):
        return np.nan
    if np.isclose(lbar, 0.0):
        return theta + (1.0 - theta) * xbar
    if np.isclose(lbar, 1.0):
        return theta * xbar
    if xbar <= lbar:
        return theta * (xbar / lbar)
    return theta + (1.0 - theta) * (xbar - lbar) / (1.0 - lbar)


def _calibrate_theta(xbars: List[float], lbar: float,
                     bounds: Tuple[float, float]) -> float:
    """Eq. ch2_calibrated: the anchor in bounds maximizing the variance of
    graded attainment over the pool, on a 0.01 grid, ties resolved toward
    the midpoint of the bounds."""
    lo, hi = float(bounds[0]), float(bounds[1])
    mid = round((lo + hi) / 2.0, 6)
    vals = [v for v in xbars if not pd.isna(v)]
    if len(vals) < 2 or pd.isna(lbar):
        return mid
    grid = np.round(np.arange(lo, hi + 1e-9, 0.01), 2)
    best_t, best_v = mid, -1.0
    for t in grid:
        v = float(np.var([_graded_degree(x, lbar, t) for x in vals]))
        if v > best_v + 1e-12 or (abs(v - best_v) <= 1e-12 and
                                  abs(t - mid) < abs(best_t - mid)):
            best_v, best_t = v, float(t)
    return float(best_t)


def _weighted_distance(a: Dict[str, float], b: Dict[str, float],
                       weights: Dict[str, float]) -> float:
    avail = [c for c in weights
             if weights[c] > 0 and not pd.isna(a.get(c)) and not pd.isna(b.get(c))]
    if not avail:
        return np.nan
    wsum = sum(weights[c] for c in avail)
    return float(sum(weights[c] * abs(a[c] - b[c]) for c in avail) / wsum)


def _fuzzy_oriented(tr: Tuple[float, float, float, float], lo: float, hi: float,
                    direction: str) -> Optional[Tuple[float, float, float, float]]:
    """Trapezoid mapped by the same oriented min max transformation as the
    crisp values; a negative direction reverses the parameter order."""
    if tr is None or any(pd.isna(v) for v in tr) or pd.isna(lo) or np.isclose(hi, lo):
        return None
    z = [(v - lo) / (hi - lo) for v in tr]
    if direction == NEGATIVE:
        z = [1.0 - v for v in reversed(z)]
    return tuple(z)


def _fuzzy_core_distance(a, b) -> float:
    """Discretized fuzzy distance (Eqs. ch2_cutdiff and ch2_discretize):
    the difference A - B has central parameters (a2 - b3, a3 - b2); their
    mean is the difference of the core midpoints; its magnitude is used."""
    d2, d3 = a[1] - b[2], a[2] - b[1]
    return abs((d2 + d3) / 2.0)


def _strictly_dominated(a: str, b: str, norm: Dict[str, Dict[str, float]],
                        criteria: List[str], min_shared: int = 1) -> bool:
    """Eq. ch2_guard: a is strictly weaker than b on every shared criterion,
    with at least min_shared shared criteria."""
    shared = [c for c in criteria
              if not pd.isna(norm[a].get(c)) and not pd.isna(norm[b].get(c))]
    if len(shared) < min_shared:
        return False
    return all(norm[a][c] < norm[b][c] for c in shared)


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
    attainment_mode: str = DEFAULT_MODE,
    theta_fixed: float = DEFAULT_THETA,
    theta_bounds: Tuple[float, float] = DEFAULT_BOUNDS,
    dominance_guard: bool = DEFAULT_GUARD,
    guard_min_shared: int = DEFAULT_QMIN,
    alternative_col: Optional[str] = None,
    fuzzy: Optional[Dict[str, Dict[str, Tuple[float, float, float, float]]]] = None,
    fuzzy_distance: bool = False,
) -> MOSDMResult:
    """
    matrix          rows are alternatives (index, or a name column given by
                    alternative_col); criteria columns hold crisp values
                    (the GMIR of a fuzzy value where the source is fuzzy),
                    NaN where unobserved.
    criteria        candidate criteria in display order; the coverage rule
                    selects the active ones.
    directions      criterion -> "positive" | "negative".
    weights         criterion -> weight, normalized over the active criteria;
                    equal weights when omitted.
    expert_limits   criterion -> threshold, highest level of the hierarchy,
                    fixed across iterations (scenario overrides enter here).
    historical_limits criterion -> quantile of the criterion's own history,
                    second level of the hierarchy, fixed across iterations.
    attainment_mode "graded_fixed" | "graded_calibrated" | "binary".
    theta_fixed     anchor of the fixed mode.
    theta_bounds    interval of the calibrated mode.
    dominance_guard on or off; guard_min_shared is q_min.
    fuzzy           optional alt -> criterion -> trapezoid (a1, a2, a3, a4)
                    used only when fuzzy_distance is on, for the within tier
                    distances; missing entries fall back to the crisp value.
    """
    if attainment_mode not in MODES:
        raise ValueError(f"unknown attainment mode {attainment_mode}")
    X = matrix.copy()
    if alternative_col is not None:
        X = X.set_index(alternative_col)
    cand = [c for c in criteria if c in X.columns]
    X = X[cand].apply(pd.to_numeric, errors="coerce")
    alts_all = [str(a) for a in X.index]
    X.index = alts_all

    # coverage rule over the initial pool
    active = [c for c in cand if int(X[c].notna().sum()) >= MIN_COVERAGE]
    inactive = [c for c in criteria if c not in active]
    w = _normalize_weights(weights or {}, active)
    # an alternative observed on no active criterion carrying positive weight
    # has no evidence under the current weights: unstratified
    weighted_active = [c for c in active if w[c] > 0]
    unstrat = [a for a in alts_all if X.loc[a, weighted_active].notna().sum() == 0] \
        if weighted_active else list(alts_all)
    alts = [a for a in alts_all if a not in unstrat]
    expert_limits = {c: float(v) for c, v in (expert_limits or {}).items()
                     if c in active and not pd.isna(v)}
    historical_limits = {c: float(v) for c, v in (historical_limits or {}).items()
                         if c in active and not pd.isna(v)}
    dirs = {c: directions.get(c, POSITIVE) for c in active}

    tiers: Dict[str, int] = {}
    scores: Dict[str, float] = {}
    targets: Dict[int, str] = {}
    iterations: List[IterationRecord] = []
    margins: Dict[int, float] = {}
    frozen_median: Dict[str, float] = {}

    pool = list(alts)
    tier_no = 0
    while pool:
        tier_no += 1
        P = X.loc[pool, active]

        # -- limits hierarchy ------------------------------------------
        limits, source = {}, {}
        for c in active:
            obs = P[c].dropna()
            if c in expert_limits:
                limits[c], source[c] = expert_limits[c], "expert"
            elif c in historical_limits:
                limits[c], source[c] = historical_limits[c], "historical"
            elif len(obs) >= MIN_COVERAGE:
                frozen_median[c] = float(obs.median())
                limits[c], source[c] = frozen_median[c], "median"
            elif c in frozen_median:
                limits[c], source[c] = frozen_median[c], "median (frozen)"
            else:
                limits[c], source[c] = (float(obs.median()) if len(obs) else np.nan), "median"

        # -- pool normalization (guard, target, distances) --------------
        norm: Dict[str, Dict[str, float]] = {a: {} for a in pool}
        flat = []
        for c in active:
            obs = P[c].dropna()
            lo, hi = (obs.min(), obs.max()) if len(obs) else (np.nan, np.nan)
            if len(obs) == 0:
                for a in pool:
                    norm[a][c] = np.nan
                flat.append(c)
                continue
            if np.isclose(lo, hi):
                flat.append(c)
            for a in pool:
                v = P.at[a, c]
                if pd.isna(v):
                    norm[a][c] = np.nan
                elif np.isclose(lo, hi):
                    norm[a][c] = 0.0
                else:
                    norm[a][c] = _orient(v, lo, hi, dirs[c])

        # -- attainment: range over the pool together with the limit -----
        thetas, lbars = {}, {}
        attain: Dict[str, Dict[str, float]] = {a: {} for a in pool}
        reached: Dict[str, List[str]] = {a: [] for a in pool}
        for c in active:
            obs = P[c].dropna()
            L = limits[c]
            if len(obs) == 0:
                for a in pool:
                    attain[a][c] = np.nan
                continue
            span = list(obs.values) + ([L] if not pd.isna(L) else [])
            lo, hi = min(span), max(span)
            graded = attainment_mode != "binary" and not pd.isna(L) and not np.isclose(lo, hi)
            if graded:
                lbars[c] = _orient(L, lo, hi, dirs[c])
                xb = {a: _orient(P.at[a, c], lo, hi, dirs[c]) for a in pool}
                if attainment_mode == "graded_calibrated":
                    thetas[c] = _calibrate_theta(list(xb.values()), lbars[c], theta_bounds)
                else:
                    thetas[c] = float(theta_fixed)
                for a in pool:
                    attain[a][c] = _graded_degree(xb[a], lbars[c], thetas[c])
            for a in pool:
                v = P.at[a, c]
                hit = _attains(v, L, dirs[c])
                if not graded:
                    attain[a][c] = hit
                if hit == 1.0:
                    reached[a].append(c)

        # -- weighted attainment -----------------------------------------
        k_w: Dict[str, float] = {}
        missing: Dict[str, List[str]] = {}
        for a in pool:
            avail = [c for c in active if not pd.isna(attain[a][c]) and w[c] > 0]
            missing[a] = [c for c in active if pd.isna(attain[a][c])]
            wsum = sum(w[c] for c in avail)
            k_w[a] = (sum(w[c] * attain[a][c] for c in avail) / wsum) if wsum > 0 else np.nan

        # an alternative with no usable criterion at this iteration cannot be
        # compared; it is deferred while comparable members remain
        comparable = [a for a in pool if not pd.isna(k_w[a])]
        if not comparable:
            comparable = list(pool)
            for a in pool:
                k_w[a] = 0.0

        # -- tolerance and candidate tier ---------------------------------
        observed_now = [c for c in active if P[c].notna().any()]
        delta = min((w[c] for c in observed_now if w[c] > 0),
                    default=1.0 / max(len(active), 1))
        k_max = max(k_w[a] for a in comparable)
        candidate = [a for a in comparable
                     if (k_max - k_w[a]) < delta - TOL_EPS or (k_max - k_w[a]) <= TOL_EPS]

        # -- dominance guard ----------------------------------------------
        demotions: List[Tuple[str, str]] = []
        members = list(candidate)
        informative = [c for c in active if c not in flat]
        if dominance_guard and len(candidate) > 1:
            for a in candidate:
                for b in candidate:
                    if a != b and _strictly_dominated(a, b, norm, informative, guard_min_shared):
                        demotions.append((a, b))
                        if a in members:
                            members.remove(a)
                        break
            if not members:
                members = list(candidate)
                demotions = []

        # -- target and within tier distances -----------------------------
        # noninformative criteria (constant over the pool) contribute nothing
        # to the target selection or to the distances
        w_inf = {c: (0.0 if c in flat else w[c]) for c in active}
        ideal = {c: (np.nan if all(pd.isna(norm[a][c]) for a in pool) else 1.0)
                 for c in active}
        dist_ideal = {a: _weighted_distance(norm[a], ideal, w_inf) for a in members}

        def _key(a, d):
            v = d[a]
            return (1, 0.0, a) if pd.isna(v) else (0, round(v / TIE_EPS) * TIE_EPS, a)

        target = min(members, key=lambda a: _key(a, dist_ideal))
        if fuzzy_distance and fuzzy:
            t_scores = {}
            for a in members:
                num = den = 0.0
                for c in active:
                    if w_inf[c] <= 0 or pd.isna(norm[a][c]) or pd.isna(norm[target][c]):
                        continue
                    obs = P[c].dropna()
                    lo, hi = obs.min(), obs.max()
                    fa = _fuzzy_oriented((fuzzy.get(a) or {}).get(c), lo, hi, dirs[c])
                    fb = _fuzzy_oriented((fuzzy.get(target) or {}).get(c), lo, hi, dirs[c])
                    d = (_fuzzy_core_distance(fa, fb) if fa and fb
                         else abs(norm[a][c] - norm[target][c]))
                    num += w_inf[c] * d
                    den += w_inf[c]
                t_scores[a] = num / den if den > 0 else np.nan
            t_scores[target] = 0.0
        else:
            t_scores = {a: _weighted_distance(norm[a], norm[target], w_inf) for a in members}
        t_scores[target] = 0.0   # S_T = 0 by definition (Eq. ch2_score)
        ordered = sorted(members, key=lambda a: _key(a, t_scores))

        for a in ordered:
            tiers[a] = tier_no
            scores[a] = t_scores[a]
        targets[tier_no] = target
        rest = [a for a in pool if a not in members]
        rest_cmp = [a for a in rest if not pd.isna(k_w.get(a, np.nan))]
        if rest_cmp:
            margins[tier_no] = float(min(k_w[a] for a in members) - max(k_w[a] for a in rest_cmp))

        iterations.append(IterationRecord(
            iteration=tier_no, pool=list(pool), limits=limits,
            limit_source=source, attainment=attain, k_weighted=k_w,
            delta=delta, candidate_tier=candidate, demotions=demotions,
            tier=ordered, target=target, scores=t_scores,
            noninformative=flat, missing=missing, thetas=dict(thetas),
            attainment_mode=attainment_mode, limit_positions=dict(lbars),
            ideal_distance=dist_ideal, reached=reached,
        ))
        pool = [a for a in pool if a not in members]

    order = sorted(alts, key=lambda a: (tiers[a],) + _key(a, scores))

    explanations = {}
    for a in alts:
        rec = iterations[tiers[a] - 1]
        n_obs = len([c for c in active if c not in rec.missing.get(a, [])])
        expl = (f"Tier {tiers[a]}: weighted attainment {rec.k_weighted[a]:.3f} "
                f"(tolerance {rec.delta:.3f}, mode {rec.attainment_mode}); reaches "
                f"{len(rec.reached[a])} of {n_obs} acceptable limits on its "
                f"observed criteria")
        if rec.missing.get(a):
            expl += f"; not observed on {len(rec.missing[a])} criterion(a), excluded"
        if a == rec.target:
            expl += "; target state of its tier (closest to the ideal point)"
        demoted_by = []
        for r in iterations[:tiers[a]]:
            demoted_by += [b for (d, b) in r.demotions if d == a]
        if demoted_by:
            expl += f"; demoted by the dominance guard (dominated by {demoted_by[0]})"
        explanations[a] = expl
    for a in unstrat:
        explanations[a] = ("Not stratifiable: observed on none of the active criteria "
                           "carrying weight; reported separately, never placed at the bottom")

    return MOSDMResult(
        alternatives=alts, criteria=active, directions=dirs, weights=w,
        tiers=tiers, order=order, scores=scores, targets=targets,
        separations=margins, iterations=iterations, explanations=explanations,
        unstratified=unstrat, inactive_criteria=inactive,
        settings={"attainment_mode": attainment_mode, "theta_fixed": theta_fixed,
                  "theta_bounds": tuple(theta_bounds), "dominance_guard": dominance_guard,
                  "guard_min_shared": guard_min_shared, "fuzzy_distance": bool(fuzzy_distance and fuzzy)},
    )
