"""Stratification core: agreement with the independent reference on random
problems, the formal properties, and the specific rules of the method."""
import numpy as np
import pandas as pd
import pytest

from modules.mosdm_core import run_mosdm, _calibrate_theta
from tests.reference import reference

P, N = "positive", "negative"


def engine(X, d, w, Lexp=None, Lhist=None, **kw):
    m, n = X.shape
    names = [f"A{i:02d}" for i in range(m)]
    crit = [f"C{j}" for j in range(n)]
    r = run_mosdm(pd.DataFrame(X, index=names, columns=crit), crit,
                  {c: (P if d[j] > 0 else N) for j, c in enumerate(crit)},
                  weights={c: w[j] for j, c in enumerate(crit)},
                  expert_limits={crit[j]: v for j, v in (Lexp or {}).items()},
                  historical_limits={crit[j]: v for j, v in (Lhist or {}).items()}, **kw)
    return r, names


def _random_case(rng):
    m, n = int(rng.integers(2, 9)), int(rng.integers(1, 7))
    X = rng.normal(size=(m, n)).round(int(rng.integers(0, 3)))
    X[rng.random((m, n)) < rng.choice([0.0, 0.15, 0.35])] = np.nan
    d = rng.choice([1, -1], n)
    w = rng.random(n) * (rng.random(n) > 0.1)
    if w.sum() == 0:
        w[0] = 1.0
    Le = {j: float(rng.normal() * 2) for j in range(n) if rng.random() < 0.25}
    Lh = {j: float(rng.normal()) for j in range(n) if rng.random() < 0.2 and j not in Le}
    mode = str(rng.choice(["binary", "graded_fixed", "graded_calibrated"]))
    guard = bool(rng.random() < 0.8)
    qmin = int(rng.choice([1, 2]))
    return X, d, w, Le, Lh, mode, guard, qmin


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_agreement_with_reference(seed):
    rng = np.random.default_rng(seed)
    bad = []
    for it in range(1500):
        X, d, w, Le, Lh, mode, guard, qmin = _random_case(rng)
        r, names = engine(X, d, w, Le, Lh, attainment_mode=mode, dominance_guard=guard,
                          guard_min_shared=qmin)
        t, S, M, T, U = reference(X, d, w, names, Le, Lh, mode=mode, guard=guard, qmin=qmin)
        ok = (r.tiers == t and r.targets == T and sorted(r.unstratified) == sorted(U)
              and all(np.isclose(r.scores[a], S[a], atol=1e-9) or (np.isnan(r.scores[a]) and np.isnan(S[a]))
                      for a in S)
              and set(r.separations) == set(M)
              and all(np.isclose(r.separations[k], M[k], atol=1e-9) for k in M))
        if not ok:
            bad.append(it)
    assert not bad, f"{len(bad)} disagreements, first cases {bad[:5]}"


def test_termination_and_completeness():
    rng = np.random.default_rng(11)
    for _ in range(800):
        X, d, w, Le, Lh, mode, guard, qmin = _random_case(rng)
        r, names = engine(X, d, w, Le, Lh, attainment_mode=mode)
        assert sorted(list(r.tiers) + r.unstratified) == sorted(names)
        assert len(r.iterations) <= len(names)
        assert sorted(r.tiers.values()) == sorted(r.tiers.values())
        assert max(r.tiers.values(), default=0) == len(r.iterations)


def test_cardinality_recovery():
    rng = np.random.default_rng(5)
    for _ in range(800):
        m, n = int(rng.integers(2, 9)), int(rng.integers(1, 6))
        X = rng.integers(0, 5, size=(m, n)).astype(float)
        d = rng.choice([1, -1], n)
        r, names = engine(X, d, np.ones(n), attainment_mode="binary")
        act = [j for j in range(n)]
        L = np.median(X, axis=0)
        kap = [sum((X[i, j] >= L[j]) if d[j] > 0 else (X[i, j] <= L[j]) for j in act) for i in range(m)]
        first = set(r.iterations[0].candidate_tier)
        assert first == {names[i] for i in range(m) if kap[i] == max(kap)}


@pytest.mark.parametrize("mode", ["graded_fixed", "graded_calibrated", "binary"])
def test_positive_affine_invariance(mode):
    rng = np.random.default_rng(7)
    for _ in range(300):
        m, n = int(rng.integers(3, 8)), int(rng.integers(1, 5))
        X = rng.normal(size=(m, n))
        d = rng.choice([1, -1], n)
        w = rng.random(n) + 0.1
        Le = {0: float(rng.normal() * 2)}
        r1, _ = engine(X, d, w, Le, attainment_mode=mode)
        a, b = rng.uniform(0.1, 5), rng.normal()
        X2 = X.copy(); X2[:, 0] = a * X2[:, 0] + b
        r2, _ = engine(X2, d, w, {0: a * Le[0] + b}, attainment_mode=mode)
        assert r1.tiers == r2.tiers and r1.targets == r2.targets
        assert all(np.isclose(r1.scores[k], r2.scores[k], atol=1e-9) for k in r1.scores)
        # negative scale with the direction reversed
        d3 = d.copy(); d3[0] = -d3[0]
        X3 = X.copy(); X3[:, 0] = -a * X3[:, 0] + b
        r3, _ = engine(X3, d3, w, {0: -a * Le[0] + b}, attainment_mode=mode)
        assert r1.tiers == r3.tiers


def test_monotonicity_in_own_performance():
    rng = np.random.default_rng(9)
    checked = 0
    for _ in range(2000):
        m, n = int(rng.integers(3, 8)), int(rng.integers(1, 5))
        X = rng.normal(size=(m, n)); d = rng.choice([1, -1], n); d[0] = 1
        lo, hi = np.sort(X[1:, 0])[[0, -1]]
        if not lo < X[0, 0] < hi:
            continue
        r1, _ = engine(X, d, np.ones(n), {0: 0.0})
        X2 = X.copy(); X2[0, 0] = rng.uniform(X[0, 0], hi)
        r2, _ = engine(X2, d, np.ones(n), {0: 0.0})
        assert r2.iterations[0].attainment["A00"]["C0"] >= r1.iterations[0].attainment["A00"]["C0"] - 1e-12
        assert r2.iterations[0].k_weighted["A00"] >= r1.iterations[0].k_weighted["A00"] - 1e-12
        checked += 1
    assert checked > 300


def test_limit_beyond_every_value_is_not_reached():
    df = pd.DataFrame({"C": [10.0, 30.0, 60.0]}, index=["X", "Y", "Z"])
    r = run_mosdm(df, ["C"], {"C": P}, expert_limits={"C": 100.0})
    a = r.iterations[0].attainment
    assert a["Z"]["C"] < 0.3 and a["Z"]["C"] == pytest.approx(0.3 * 50 / 90)
    assert "reaches 0 of 1" in r.explanations["Z"]
    r = run_mosdm(df, ["C"], {"C": P}, expert_limits={"C": 0.0})
    a = r.iterations[0].attainment
    assert a["X"]["C"] > 0.3 and a["X"]["C"] == pytest.approx(0.3 + 0.7 * 10 / 60)


def test_alternative_without_evidence_is_unstratified():
    df = pd.DataFrame({"C1": [1, 5, np.nan], "C2": [2, 1, np.nan]}, index=["X", "Y", "Z"], dtype=float)
    r = run_mosdm(df, ["C1", "C2"], {"C1": P, "C2": P})
    assert r.unstratified == ["Z"] and "Z" not in r.tiers
    assert "Not stratifiable" in r.explanations["Z"]


def test_coverage_rule_and_frozen_median():
    df = pd.DataFrame({"C1": [9, 1, 2, 0.5], "C2": [9, np.nan, np.nan, 1], "C3": [1, np.nan, np.nan, np.nan]},
                      index=["A", "B", "C", "D"], dtype=float)
    r = run_mosdm(df, ["C1", "C2", "C3"], {"C1": P, "C2": P, "C3": P})
    assert r.inactive_criteria == ["C3"] and "C3" not in r.criteria
    second = r.iterations[1]
    assert second.limit_source["C2"] == "median (frozen)" and second.limits["C2"] == pytest.approx(5.0)
    assert second.attainment["D"]["C2"] == 0.0   # 1 < frozen median 5, not judged against itself


def test_mid_ranks_for_ties_and_zero_target_distance():
    df = pd.DataFrame({"C": [3.0, 3.0, 3.0, 1.0]}, index=["P", "Q", "R", "S"])
    r = run_mosdm(df, ["C"], {"C": P})
    rk = r.ranks()
    assert rk["P"] == rk["Q"] == rk["R"] == 2.0 and rk["S"] == 4.0
    assert all(r.scores[r.targets[t]] == 0.0 for t in r.targets)


def test_calibrated_ties_resolve_to_the_midpoint():
    assert _calibrate_theta([0.2, 0.2], 0.5, (0.6, 0.9)) == pytest.approx(0.75)
    assert _calibrate_theta([0.2], 0.5, (0.3, 0.7)) == pytest.approx(0.5)


def test_constant_criterion_is_noninformative_for_guard_and_distances():
    df = pd.DataFrame({"C1": [5.0, 1.0], "C2": [2.0, 2.0]}, index=["A", "B"])
    r = run_mosdm(df, ["C1", "C2"], {"C1": P, "C2": P}, weights={"C1": 0.1, "C2": 0.9})
    rec = r.iterations[0]
    assert "C2" in rec.noninformative
    # B is strictly weaker on the only informative criterion: demoted if it is in the candidate tier
    assert r.tiers["A"] < r.tiers["B"]


def test_fuzzy_distance_equals_crisp_for_intervals():
    df = pd.DataFrame({"C1": [1.5, 3.0, 2.0], "C2": [2.0, 2.5, 4.0]}, index=["A", "B", "C"])
    fz = {"A": {"C1": (1, 1, 2, 2)}, "B": {"C1": (2, 2, 4, 4)}, "C": {"C1": (2, 2, 2, 2)}}
    r1 = run_mosdm(df, ["C1", "C2"], {"C1": P, "C2": P}, weights={"C1": 0.2, "C2": 0.2})
    r2 = run_mosdm(df, ["C1", "C2"], {"C1": P, "C2": P}, weights={"C1": 0.2, "C2": 0.2},
                   fuzzy=fz, fuzzy_distance=True)
    assert r1.tiers == r2.tiers
    assert all(np.isclose(r1.scores[a], r2.scores[a]) for a in r1.scores)


def test_guard_ignores_noninformative_criteria():
    df = pd.DataFrame({"C1": [5.0, 4.0, 1.0], "C2": [2.0, 2.0, 2.0]}, index=["A", "B", "C"])
    r = run_mosdm(df, ["C1", "C2"], {"C1": P, "C2": P}, weights={"C1": 0.3, "C2": 0.7})
    rec = r.iterations[0]
    assert set(rec.candidate_tier) == {"A", "B"}
    assert ("B", "A") in rec.demotions and r.tiers["A"] == 1 and r.tiers["B"] == 2
    assert 0 < r.separations[1] < rec.delta   # a boundary inside the tolerance band: imposed by the guard
