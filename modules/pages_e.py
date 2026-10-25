"""
pages_e.py

Overview, Sensitivity & Simulation, External Evidence & Data, and the
Elicitation Studio.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules import ui, state, controls, feeds
from modules import scenario_engine as se
from modules import ssdm_core as sc
from modules import sensitivity as sens
from modules import weight_manager as wm
from modules.agent_engine import AGENTS, CORE4, default_limits, active_criteria, profile
from modules.load_data import load_citations, load_derivations, load_data_dictionary


def _strip(jurisdiction):
    s = state.settings()
    from modules import live_subcriteria as ls
    ref = ls.reference_week(jurisdiction)
    ui.context_strip([
        ("Scope", "US National" if jurisdiction == "National" else "Michigan", ""),
        ("Data week", str(ref)[:10] if ref is not None else "none", ""),
        ("Attainment", s["attainment_mode"].replace("_", " "), ""),
        ("lambda, exponents", f"{s['lambda']:g}; {s['alpha']:g}, {s['beta']:g}, {s['gamma']:g}", ""),
        ("Limits window", "since May 2023" if s["history_window"] != "full" else "full history", ""),
    ])


def overview(jurisdiction="National"):
    _strip(jurisdiction)
    lam, a, b, g = controls.params()
    theta = float(state.setting("theta_si"))
    scope = "National" if jurisdiction == "National" else "Michigan"
    st.title(f"{scope} Prioritization Overview")
    st.caption("Scenario weighted and integrated over eight perspectives. Every number traces to "
               "its source, its calculation and its audit trail (Method Trace).")
    with st.spinner("Computing the integrated picture"):
        M, res, _ = sc.cross_agent_run(jurisdiction, lam, a, b, g, "", False)
        _, base, _ = sc.cross_agent_run(jurisdiction, lam, a, b, g, "", True)
        quads = {k: sc.quadrants(sc.indicators(k, jurisdiction, lam, a, b, g), theta) for k in AGENTS}
        prob = se.probability_table(lam, a, b, g)
    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Final integrated stratification")
        rk = res.ranks()
        for d in res.order:
            t = res.tiers[d]
            cls = {k: q.set_index("Disease").at[d, "Quadrant"] for k, q in quads.items()
                   if d in set(q["Disease"])}
            crit = [sc.AGENT_LABELS[k] for k, c in cls.items() if c == "Conditionally critical"]
            struct = [sc.AGENT_LABELS[k] for k, c in cls.items() if c == "Structural priority"]
            why = (f"Rank {rk[d]:g}; baseline tier {base.tiers.get(d)}; structural priority in "
                   f"{len(struct)} perspective(s)" + (f"; conditionally critical in {', '.join(crit)}" if crit else ""))
            ui.tier_card(t, ui.SHORT.get(d, d), [], why, is_target=(res.targets[t] == d))
    with right:
        st.subheader("Typology across perspectives")
        tab = pd.DataFrame({sc.AGENT_LABELS[k]: q.set_index("Disease")["Quadrant"] for k, q in quads.items()})
        tab.index = [ui.SHORT.get(i, i) for i in tab.index]
        st.dataframe(tab, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Scenario weights")
        st.plotly_chart(ui.prob_bars(prob), width="stretch")
    with c2:
        st.subheader("Signals")
        from modules import evidence_gate as eg, live_subcriteria as ls
        adm = se.admission_table()
        diff = adm[(adm["Source"] == "computed from the weekly series")
                   & (adm["n_s"] != pd.to_numeric(adm["Registry n_s"], errors="coerce"))]
        if not diff.empty:
            ui.signal("<b>Admission.</b> Recomputed episode counts differ from the curated registry for "
                      + ", ".join(diff["Scenario ID"]) + " (Scenarios page).", warn=True)
        ext = eg.external_evidence()
        if not ext.empty and "differs_notably" in ext.columns:
            n = int(ext["differs_notably"].astype(bool).sum())
            ui.signal(f"<b>External evidence.</b> {len(ext)} validated figure(s); {n} differ notably from "
                      "the dashboard's values (information only).", warn=n > 0)
        defaults = [k for k in AGENTS if wm.get_entry("criterion", k) is None]
        if defaults:
            ui.signal(f"<b>Judgment.</b> Equal default weights on {len(defaults)} of 8 perspectives; "
                      "expert input can be entered in the Elicitation Studio.")
        notes = ls.freshness_report(jurisdiction)
        from modules.agents.agent7_outbreak import rt_current
        n_stale = len(notes) + len(rt_current(jurisdiction)[1])
        if n_stale:
            ui.signal(f"<b>Surveillance.</b> {n_stale} live value(s) treated as missing because stale "
                      "or out of scope (agent pages list them).", warn=True)


def sensitivity_page(jurisdiction="National"):
    _strip(jurisdiction)
    lam, a, b, g = controls.params()
    st.title("Sensitivity & Simulation")
    st.caption("Recomputes the full chain with the current inputs and settings. The full evaluation "
               "protocol of the paper (ablation, factorial design, Sobol indices, acceptability, "
               "benchmarks, imperfect evidence, replay, scaling) runs from scripts/experiments.")
    st.subheader("1) One at a time sweeps")
    param = st.selectbox("Parameter", ["lam", "a", "b", "g"],
                         format_func={"lam": "lambda", "a": "alpha", "b": "beta", "g": "gamma"}.get)
    grid = tuple(np.round(np.arange(0.0, 1.01, 0.1), 2)) if param == "lam" else \
        tuple(np.round(np.arange(0.0, 2.01, 0.25), 2))
    ak = st.selectbox("Agent for the expected ranks", list(AGENTS), format_func=lambda k: AGENTS[k]["title"],
                      key="sens_agent")
    if not st.session_state.get("sens_run") and not st.button("Run the sweeps", type="primary"):
        st.caption("Each sweep re runs the complete chain at every grid point; results are kept "
                   "until an input or setting changes.")
        return
    st.session_state["sens_run"] = True
    with st.spinner("Sweeping the full chain"):
        sw = sens.parameter_sweep(param, grid, jurisdiction, lam, a, b, g)
        ps = sens.prob_sweep(param, grid, lam, a, b, g)
        rs = sens.r_sweep(param, grid, ak, jurisdiction, lam, a, b, g)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for sid in sorted(ps["Scenario"].unique()):
            dd = ps[ps["Scenario"] == sid]
            fig.add_trace(go.Scatter(x=dd["value"], y=dd["p_s"], mode="lines", name=sid))
        fig.update_yaxes(title="Scenario weight p_s"); fig.update_xaxes(title=param)
        st.plotly_chart(ui._base_layout(fig), width="stretch")
    with c2:
        fig = go.Figure()
        for d in CORE4:
            dd = rs[rs["Disease"] == d]
            fig.add_trace(go.Scatter(x=dd["value"], y=dd["R"], mode="lines+markers", name=ui.SHORT.get(d, d)))
        fig.update_yaxes(title="Expected rank R", autorange="reversed"); fig.update_xaxes(title=param)
        st.plotly_chart(ui._base_layout(fig), width="stretch")
    fig = go.Figure()
    for d in CORE4:
        dd = sw[sw["Disease"] == d]
        fig.add_trace(go.Scatter(x=dd["value"], y=dd["Final rank"], mode="lines+markers", name=ui.SHORT.get(d, d)))
    fig.update_yaxes(title="Final integrated rank", autorange="reversed"); fig.update_xaxes(title=param)
    st.plotly_chart(ui._base_layout(fig), width="stretch")
    st.caption("lambda moves nothing until a scenario holds both expert and AI ratings.")

    st.subheader("2) Monte Carlo acceptability of the perspective weights")
    n = st.slider("Draws", 100, 2000, 500, 100)
    conc = st.slider("Concentration (higher = smaller perturbations)", 10, 200, 60, 10)
    with st.spinner("Simulating"):
        tiers, ranks = sens.weight_monte_carlo(jurisdiction, lam, a, b, g, "", n, float(conc))
    c1, c2 = st.columns(2)
    c1.dataframe(tiers.round(3), width="stretch", hide_index=True)
    fig = go.Figure()
    for d in CORE4:
        fig.add_trace(go.Box(y=ranks[ranks["Disease"] == d]["Rank"], name=ui.SHORT.get(d, d), boxmean=True))
    fig.update_yaxes(title="Integrated rank", autorange="reversed")
    c2.plotly_chart(ui._base_layout(fig), width="stretch")

    st.subheader("3) Stability threshold and attainment mode")
    c3, c4 = st.columns(2)
    with c3:
        ak2 = st.selectbox("Agent for the threshold sweep", list(AGENTS), format_func=lambda k: AGENTS[k]["title"],
                           key="sens_theta_agent")
        ts = sens.theta_sweep(ak2, jurisdiction, lam, a, b, g)
        piv = ts.pivot_table(index="Disease", columns="theta", values="Quadrant", aggfunc="first")
        piv.index = [ui.SHORT.get(i, i) for i in piv.index]
        st.dataframe(piv, width="stretch")
    with c4:
        mc = sens.mode_comparison(jurisdiction, lam, a, b, g)
        mc.index = [ui.SHORT.get(i, i) for i in mc.index]
        st.dataframe(mc, width="stretch")


def evidence_data(jurisdiction="National"):
    _strip(jurisdiction)
    st.title("External Evidence & Data")
    tabs = st.tabs(["External evidence", "Gate ledger", "Surveillance feeds", "Decision matrix",
                    "Citations", "Derivations", "Scenario registry", "Data dictionary"])
    from modules import evidence_gate as eg
    with tabs[0]:
        ext = eg.external_evidence()
        if ext.empty:
            st.info("No validated external evidence yet; retrieve it on any agent page (editor).")
        else:
            st.dataframe(ext.astype(str), width="stretch", hide_index=True)
            st.caption("Validated figures from official external sources, for decision makers. They "
                       "never change the stratification; 'differs_notably' marks a figure more than 25 "
                       "percent away from the dashboard's value.")
    with tabs[1]:
        led = eg.load_ledger()
        if led.empty:
            st.info("No gate runs recorded yet.")
        else:
            f = st.multiselect("Status", sorted(led["status"].unique()), default=sorted(led["status"].unique()))
            st.dataframe(led[led["status"].isin(f)].astype(str), width="stretch", hide_index=True)
            st.download_button("Download ledger (CSV)", led.to_csv(index=False), "evidence_ledger.csv")
    with tabs[2]:
        st.dataframe(feeds.data_status(), width="stretch", hide_index=True)
        log = feeds.update_log()
        if log is not None:
            st.markdown("**What changed with the latest weekly update**")
            st.dataframe(log.astype(str), width="stretch", hide_index=True) if not log.empty else \
                st.caption("No change in episode counts, scenario weights, limits or final tiers.")
        from modules import live_subcriteria as ls
        st.markdown("**Historical statistics inside the limits window**")
        st.dataframe(pd.DataFrame(ls.historical_stats(jurisdiction, 1, state.setting("history_window"))).T,
                     width="stretch")
    with tabs[3]:
        from modules.agent_m_data import get_active_decision_matrix
        st.dataframe(get_active_decision_matrix(jurisdiction).astype(str), width="stretch", hide_index=True)
    with tabs[4]:
        st.dataframe(load_citations().astype(str), width="stretch", hide_index=True)
    with tabs[5]:
        st.dataframe(load_derivations().astype(str), width="stretch", hide_index=True)
    with tabs[6]:
        st.dataframe(se.get_registry().astype(str), width="stretch", hide_index=True)
    with tabs[7]:
        st.dataframe(load_data_dictionary().astype(str), width="stretch", hide_index=True)


def elicitation_studio(jurisdiction="National"):
    _strip(jurisdiction)
    st.title("Elicitation Studio")
    st.caption("Every judgment input: baseline weights and limits per perspective, weights under each "
               "scenario, the perspective weights W_g, and (on the Scenarios page) severity and impact. "
               "Each input keeps its provenance, author and time.")
    rows = []
    for k in AGENTS:
        _, wprov = wm.get_weights(k, active_criteria(profile(k, jurisdiction, tuple(CORE4)), k))
        _, lprov = wm.get_expert_limits(k)
        scen = [s for s in se.scenario_ids(True) if wm.get_entry("criterion", f"{k}::{s}")]
        rows.append({"Perspective": AGENTS[k]["title"], "Baseline weights": wprov, "Limits": lprov,
                     "Scenario specific weights": ", ".join(scen) or "none (data derived emphasis)"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(f"Perspective weights W_g: {sc.get_agent_weights()[1]}.")

    ak = st.selectbox("Perspective", list(AGENTS), format_func=lambda k: AGENTS[k]["title"], key="studio_agent")
    prof = profile(ak, jurisdiction, tuple(CORE4))
    act = active_criteria(prof, ak)
    t1, t2, t3 = st.tabs(["Baseline weights", "Acceptable limits", "Weights under a scenario"])
    with t1:
        wm.render_weight_elicitation(ak, AGENTS[ak]["title"], act)
    with t2:
        wm.render_limits_editor(ak, AGENTS[ak]["title"], AGENTS[ak]["criteria"],
                                default_limits(ak, jurisdiction, tuple(CORE4)))
    with t3:
        sid = st.selectbox("Scenario", se.scenario_ids(True), key="studio_scen")
        w, prov = se.scenario_weights(sid, ak, act)
        st.caption(f"Current weights under {sid} ({prov}).")
        wm.render_weight_elicitation(f"{ak}::{sid}", f"{AGENTS[ak]['title']} under {sid}", act)
    st.subheader("Perspective weights W_g")
    wm.render_weight_elicitation("agent_level::global", "Cross agent integration", list(AGENTS))
