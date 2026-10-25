"""Scenario layer: episode rule, elevation ratio, interval, probability rule."""
import numpy as np
import pandas as pd
import pytest

from modules import scenario_admission as sa
from modules.weight_manager import aggregate_trapezoids, gmir, INTENSITY_SCALE, IMPORTANCE_SCALE


def flags(pattern, start="2024-01-06"):
    idx = pd.date_range(start, periods=len(pattern), freq="W-SAT")
    return pd.Series([c == "1" for c in pattern], index=idx)


def test_episode_rule():
    f = flags("0110001000011100001")
    assert len(sa.episodes_from_flags(f, min_weeks=1, gap_weeks=0)) == 4
    # gaps of three weeks or fewer join; isolated single weeks need a partner
    eps = sa.episodes_from_flags(f, min_weeks=2, gap_weeks=3)
    assert [e[2] for e in eps] == [3, 3]
    # S4 style: runs shorter than three consecutive weeks are not episodes
    eps = sa.episodes_from_flags(flags("0110111101110"), min_weeks=3, gap_weeks=0, run_min=3)
    assert [e[2] for e in eps] == [4, 3]


def test_episode_breaks_at_missing_weeks():
    f = flags("111")
    f = pd.concat([f, pd.Series([True], index=[f.index[-1] + pd.Timedelta(weeks=6)])])
    assert len(sa.episodes_from_flags(f, min_weeks=1, gap_weeks=0)) == 2
    assert len(sa.episodes_from_flags(f, min_weeks=1, gap_weeks=5)) == 1


def test_elevation_ratio():
    idx = pd.date_range("2024-01-06", periods=10, freq="W-SAT")
    s = pd.Series([1, 1, 1, 4, 4, 4, 1, 1, 1, 1], index=idx, dtype=float)
    m = sa.MarkerResult("X", s > 2, [(idx[3], idx[5], 3)], (idx[0], idx[-1]), "")
    r, ne, nw = sa.elevation_ratio(s, m)
    assert r == pytest.approx(4.0) and ne == 3 and nw == 10
    neg = s - 2
    assert np.isnan(sa.elevation_ratio(neg, m)[0])


def test_poisson_interval():
    from modules.scenario_engine import poisson_rate_interval
    lo, hi = poisson_rate_interval(4, 2.0)
    assert lo == pytest.approx(0.5449, abs=1e-3) and hi == pytest.approx(5.1209, abs=1e-3)
    assert poisson_rate_interval(0, 1.0)[0] == 0.0


def test_aggregation_rules():
    t = [IMPORTANCE_SCALE["Low"], IMPORTANCE_SCALE["High"]]
    mean = aggregate_trapezoids(t, "mean")
    assert gmir(*mean) == pytest.approx((gmir(*t[0]) + gmir(*t[1])) / 2)
    mm = aggregate_trapezoids(t, "minmax_geometric")
    assert mm[0] == 0.15 and mm[3] == 1.0
    assert mm[1] == pytest.approx(np.sqrt(0.30 * 0.80)) and mm[2] == pytest.approx(np.sqrt(0.40 * 0.90))


def test_scale_values_match_the_chapter():
    want = {"Very low": 0.075, "Low": 0.350, "Medium": 0.600, "High": 0.842, "Very high": 0.958}
    for k, v in want.items():
        assert gmir(*IMPORTANCE_SCALE[k]) == pytest.approx(v, abs=5e-4)
    want = {"Negligible": 0.067, "Moderate": 0.400, "Severe": 0.700, "Critical": 0.933}
    for k, v in want.items():
        assert gmir(*INTENSITY_SCALE[k]) == pytest.approx(v, abs=5e-4)
