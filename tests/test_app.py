"""Every page of the app and the interactive flows (skipped without feeds)."""
import os
from pathlib import Path

import pytest

from modules import feeds, state

ROOT = Path(__file__).resolve().parents[1]
HAVE = (feeds.data_dir() / "feeds" / "Weekly Hospital Respiratory Dat.parquet").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="no surveillance feeds available")
PAGES = ["Overview", "Domain Agents", "Scenarios & Probability", "SSDM Integration",
         "Sensitivity & Simulation", "Method Trace", "External Evidence & Data",
         "Elicitation Studio", "Method Settings", "Methods & About"]


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    state.set_backend(state.JSONBackend(tmp_path / "inputs.json"))
    state._GLOBAL_SESSION.clear()
    yield
    state.set_backend(state.MemoryBackend())


def app(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=900)
    at.run()
    at.sidebar.radio(key="nav_page").set_value(page).run()
    return at


def ok(at):
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert not at.error, [str(e.value)[:300] for e in at.error]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    ok(app(page))


def test_every_agent_scenario_and_mode():
    at = app("Domain Agents")
    from modules.agent_engine import AGENTS
    from modules import scenario_engine as se
    for ak in AGENTS:
        at.selectbox(key="nav_agent").set_value(ak).run()
        for sc in ["Baseline"] + se.scenario_ids(True):
            at.selectbox(key=f"{ak}_scen").set_value(sc).run()
            ok(at)
    for md in ["graded_calibrated", "binary", "graded_fixed"]:
        state.set_session_setting("attainment_mode", md)
        at.run()
        ok(at)


def _strat_tables(at):
    return [d.value for d in at.dataframe if "Overall Rank" in getattr(d.value, "columns", [])]


def test_disease_selection_reaches_scenarios():
    at = app("Domain Agents")
    ms = at.multiselect(key="A1_sel")
    ms.set_value(ms.value[:3]).run()
    at.selectbox(key="A1_scen").set_value("S1").run()
    ok(at)
    assert len(_strat_tables(at)[0]) == 3


def test_questionnaire_respondents_apply_immediately():
    at = app("Elicitation Studio")
    at.radio(key="wm_mode_A1").set_value("Questionnaire (linguistic)").run()
    at.text_input(key="wm_qname_A1").set_value("expert one").run()
    sl = [s for s in at.select_slider if s.key and s.key.startswith("wm_q_A1_")]
    sl[0].set_value("Very high").run()
    at.button(key="wm_qs_A1").click().run()
    at.text_input(key="wm_qname_A1").set_value("expert two").run()
    at.button(key="wm_qs_A1").click().run()
    at.button(key="wm_qb_A1").click().run()
    ok(at)
    e = state.get("weights", "A1")
    assert e["provenance"].startswith("questionnaire (2 respondents")
    caps = [c.value for c in at.caption if c.value.startswith("Active weights")]
    assert any("questionnaire" in c for c in caps)


def test_expert_rating_reaches_the_integration():
    at = app("SSDM Integration")
    before = [d.value for d in at.dataframe if "R (expected rank)" in getattr(d.value, "columns", [])][0]
    from modules import scenario_engine as se
    se.set_expert_terms("S4", "Negligible", "Negligible", "e1")
    at.run()
    after = [d.value for d in at.dataframe if "R (expected rank)" in getattr(d.value, "columns", [])][0]
    assert not before.equals(after)


def test_limits_editor_shows_the_real_defaults():
    at = app("Elicitation Studio")
    t = [d.value for d in at.dataframe if "Default source" in getattr(d.value, "columns", [])][0]
    src = dict(zip(t["Subcriterion"], t["Default source"]))
    assert src["Current Weekly Hospitalization Rate (per 100k)"].startswith("historical median")
    assert src["Annual Incidence Rate (per 100k)"] == "pool median"


def test_viewer_cannot_edit(monkeypatch):
    monkeypatch.setenv("EDITOR_PASSWORD", "pw")
    monkeypatch.delenv("SSDM_EDITOR", raising=False)
    at = app("Elicitation Studio")
    ok(at)
    assert not [b for b in at.button if b.key and b.key.startswith("wm_")]
    assert any("Read only" in i.value for i in at.info)
    at.sidebar.text_input(key="login_name").set_value("Shayan").run()
    at.sidebar.text_input(key="login_pw").set_value("pw").run()
    [b for b in at.sidebar.button if b.label == "Sign in"][0].click().run()
    ok(at)
    assert [b for b in at.button if b.key and b.key.startswith("wm_")]


def test_settings_page_changes_the_run():
    at = app("Method Settings")
    at.selectbox[0].set_value("binary").run()
    [b for b in at.button if b.label == "Apply to my session"][0].click().run()
    ok(at)
    assert state.setting("attainment_mode") == "binary" or \
        at.session_state["_ssdm_session_settings"].get("attainment_mode") == "binary"


def test_settings_reach_the_engine():
    at = app("Method Settings")
    at.selectbox[0].set_value("binary").run()
    [b for b in at.button if b.label == "Apply to my session"][0].click().run()
    at.sidebar.radio(key="nav_page").set_value("Domain Agents").run()
    ok(at)
    assert any("binary" in c.value.lower() for c in at.caption)
    audits = [d.value for d in at.dataframe if "Anchor theta" in getattr(d.value, "columns", [])]
    assert audits and all(str(v) in ("None", "nan") for v in audits[0]["Anchor theta"])
