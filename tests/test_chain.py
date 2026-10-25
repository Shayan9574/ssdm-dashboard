"""The full chain on real surveillance data (skipped when no feeds are
available): every scenario and integration quantity recomputed
independently from the engine's own per scenario results and inputs."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules import feeds, state

HAVE = (feeds.data_dir() / "feeds" / "Weekly Hospital Respiratory Dat.parquet").exists() \
    or feeds.workbook_path() is not None
pytestmark = pytest.mark.skipif(not HAVE, reason="no surveillance feeds available")


@pytest.fixture(autouse=True)
def fresh_store(tmp_path):
    state.set_backend(state.JSONBackend(tmp_path / "inputs.json"))
    state._GLOBAL_SESSION.clear()
    yield
    state.set_backend(state.MemoryBackend())


def _gm(t):
    return (t[0] + 2 * t[1] + 2 * t[2] + t[3]) / 6


def test_history_window_and_limits():
    from modules import live_subcriteria as ls
    ref = ls.reference_week()
    assert ls.window_start(ref) == max(ls.PHE_END, pd.Timestamp(year=ls.season_start(ref).year - 5, month=10, day=1))
    hv = ls.historical_values()
    hs = ls.historical_stats()
    for c, st_ in hs.items():
        assert st_["p50"] == pytest.approx(float(hv[c].median()))
        assert st_["p90"] == pytest.approx(float(hv[c].quantile(0.9)))
        assert pd.to_datetime(hv[c].index).min() >= ls.window_start(ref)


def test_scenario_weights_and_tightening():
    from modules import scenario_engine as se, live_subcriteria as ls
    from modules.agent_engine import AGENTS, CORE4, profile, active_criteria
    from modules import weight_manager as wm
    hv = ls.historical_values()
    for sid in se.scenario_ids(True):
        em = se.emphasis_multipliers(sid)
        assert (em["Multiplier k"] >= 1.0).all() if not em.empty else True
        for k in AGENTS:
            act = active_criteria(profile(k, "National", tuple(CORE4)), k)
            w, prov = se.scenario_weights(sid, k, act)
            base, _ = wm.get_weights(k, act)
            kk = {r["Criterion"]: r["Multiplier k"] for _, r in em[em["Agent"] == k].iterrows()} if not em.empty else {}
            raw = {c: base[c] * kk.get(c, 1.0) for c in act}
            tot = sum(raw.values())
            assert all(w[c] == pytest.approx(raw[c] / tot) for c in act)
            ov = se.scenario_limit_overrides(sid, k)
            for c, v in ov.items():
                if c == se.RT_CRITERION:
                    assert v == 1.0
                else:
                    assert c in se.emphasized(sid, k)
                    assert v == pytest.approx(float(hv[c].quantile(0.9)))


def test_probabilities_indicators_stability_integration():
    from modules import scenario_engine as se, ssdm_core as sc
    from modules.weight_manager import INTENSITY_SCALE as IS
    from modules.agent_engine import AGENTS
    se.set_expert_terms("S1", "Severe", "Critical", "e1")
    se.set_expert_terms("S1", "Moderate", "Severe", "e2")
    se.set_ai_assessment("S1", "Moderate", "Severe", "", "m")
    se.set_ai_assessment("S8", "Critical", "Critical", "", "m")
    lam, al, be, ga = 0.7, 1.0, 1.3, 0.6
    adm = se.admission_table().set_index("Scenario ID")
    pi = {}
    for sid, r in adm[adm["weighted"]].iterrows():
        a = se.get_assessments(sid)

        def v(dim):
            ex = [IS[t[dim]] for t in a.get("expert_respondents", {}).values()]
            ev = _gm(np.mean(np.array(ex), axis=0)) if ex else None
            av = _gm(IS[a["ai"][dim]]) if a.get("ai") else None
            if ev is not None and av is not None:
                return lam * ev + (1 - lam) * av
            return ev if ev is not None else (av if av is not None else _gm(IS["Moderate"]))
        pi[sid] = (r["n_s"] / r["Y_s"]) ** al * v("severity") ** be * v("impact") ** ga
    tot = sum(pi.values())
    p = se.probability_map(lam, al, be, ga)
    assert set(p) == set(pi)
    assert all(p[s] == pytest.approx(pi[s] / tot, rel=1e-12) for s in pi)
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-12)

    for k in AGENTS:
        base, _ = se.run_scenario(k, None)
        runs = {s: se.run_scenario(k, s)[0] for s in p}
        ind = sc.indicators(k, "National", lam, al, be, ga).set_index("Disease")
        stab = se.stability_table(k, "National", lam, al, be, ga).set_index("Disease")
        m = len(base.order)
        for d in base.order:
            rk = {s: runs[s].ranks()[d] for s in p}
            R = sum(p[s] * rk[s] for s in p)
            Phi = sum(p[s] * (runs[s].tiers[d] == 1) for s in p)
            sig = np.sqrt(sum(p[s] * (rk[s] - R) ** 2 for s in p))
            SI = max(0.0, 1 - 2 * sig / (m - 1))
            assert ind.at[d, "R (expected rank)"] == pytest.approx(R, abs=1e-12)
            assert ind.at[d, "Phi (top tier prob.)"] == pytest.approx(Phi, abs=1e-12)
            assert ind.at[d, "SI (stability)"] == pytest.approx(SI, abs=1e-12)
            delta = {s: runs[s].tiers[d] - base.tiers[d] for s in p}
            assert stab.at[d, "Exposure E"] == pytest.approx(sum(p[s] for s in p if delta[s] != 0), abs=1e-12)
            assert stab.at[d, "Instability I"] == pytest.approx(sum(p[s] * abs(delta[s]) for s in p), abs=1e-12)

    M = sc.second_order("National", lam, al, be, ga)
    for k in AGENTS:
        ind = sc.indicators(k, "National", lam, al, be, ga).set_index("Disease")
        col = M[sc.AGENT_LABELS[k]]
        m = int(ind["R (expected rank)"].notna().sum())
        for d in col.index:
            want = ind.at[d, "R (expected rank)"]
            if k in sc.INTEGRATION_INVERT:
                want = m + 1 - want
            assert col[d] == pytest.approx(want, abs=1e-12)


def test_inputs_reach_every_result():
    from modules import scenario_engine as se, ssdm_core as sc
    a = sc.indicators("A1", "National", 0.7, 1, 1, 1)
    se.set_expert_terms("S4", "Negligible", "Negligible", "e1")
    b = sc.indicators("A1", "National", 0.7, 1, 1, 1)
    assert not a.equals(b)
    _, r1, _ = sc.cross_agent_run("National", 0.7, 1, 1, 1)
    state.put("weights", "agent_level::global",
              {"values": {k: (5.0 if k == "A4" else 1.0) for k in sc.AGENTS}}, provenance="direct selection")
    _, r2, wp = sc.cross_agent_run("National", 0.7, 1, 1, 1)
    assert wp == "direct selection"
    assert r2.weights[sc.AGENT_LABELS["A4"]] == pytest.approx(5 / 12)


def test_selected_diseases_reach_scenario_runs():
    from modules import scenario_engine as se
    sel = ("Influenza", "COVID-19", "Respiratory Syncytial Virus (RSV)")
    for sid in [None, "S1", "S4"]:
        r, _ = se.run_scenario("A1", sid, "National", sel)
        assert set(r.order) | set(r.unstratified) == set(sel)


def test_stale_live_values_are_missing():
    from modules.agents.agent7_outbreak import rt_current
    from modules import live_subcriteria as ls
    cur, notes = rt_current("National")
    ref = ls.reference_week("National")
    for k, rec in cur.items():
        assert (ref - pd.Timestamp(rec["as_of_date"])).days <= 7 * ls.MAX_AGE_WEEKS
    for n in notes:
        assert n["Age (weeks)"] > ls.MAX_AGE_WEEKS


def test_scenario_tightening_never_loosens_an_expert_limit():
    from modules import scenario_engine as se
    from modules.agent_engine import environment, CORE4
    c = "Current Weekly Hospitalization Rate (per 100k)"
    ov = se.scenario_limit_overrides("S1", "A1")
    assert c in ov
    state.put("limits", "A1", {"values": {c: ov[c] * 3}}, provenance="expert")
    _, _, lim, _ = environment("A1", "National", tuple(CORE4), "S1")
    assert lim[c] == pytest.approx(ov[c] * 3)
    state.put("limits", "A1", {"values": {c: ov[c] / 3}}, provenance="expert")
    _, _, lim, _ = environment("A1", "National", tuple(CORE4), "S1")
    assert lim[c] == pytest.approx(ov[c])


def test_replay_runs_and_zero_episode_scenarios_get_zero_weight():
    from modules import live_connectors as lc, scenario_engine as se
    try:
        lc.AS_OF = pd.Timestamp("2024-03-02")
        adm = se.admission_table()
        p = se.probability_map()
        zero = adm[adm["weighted"] & (adm["n_s"] == 0)]["Scenario ID"].tolist()
        assert all(s not in p for s in zero)
        assert sum(p.values()) == pytest.approx(1.0)
    finally:
        lc.AS_OF = None
