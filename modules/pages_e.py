"""
pages_e.py

Checkpoint E pages: Overview (the landing page leading with the
answer), Sensitivity & Simulation, Evidence & Data, and the
Elicitation Studio. Built on the existing engines; heavy computation
flows through the same cached functions every page shares, so the
first load computes once and page switches are instant.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules import ui
from modules import scenario_engine as se
from modules import ssdm_core as sc
from modules import sensitivity as sens
from modules import weight_manager as wm
from modules.agent_engine import AGENTS, CORE4, run_agent
from modules.evidence_gate import load_ledger
from modules.load_data import (load_citations, load_derivations,
                               load_data_dictionary, load_scenario_registry)


def _params():
    return (st.session_state.get("g_lam", 0.7),
            st.session_state.get("g_alpha", 1.0),
            st.session_state.get("g_beta", 1.0),
            st.session_state.get("g_gamma", 1.0),
            st.session_state.get("attainment_mode", "graded_calibrated"))


def _strip(jurisdiction):
    lam, a, b, g, mode = _params()
    led = load_ledger()
    n_ev = 0 if led.empty else int((led["status"] == "validated").sum())
    ov_on = st.session_state.get("use_evidence_overlay", True)
    ui.context_strip([
        ("Scope", "US National" if jurisdiction == "National" else "Michigan", ""),
        ("Engine", {"graded_calibrated": "2nd gen MOSDM, calibrated",
                    "graded_fixed": "2nd gen MOSDM, fixed anchor",
                    "binary": "1st gen (ablation)"}[mode], ""),
        ("lambda, exponents", f"{lam:g}; {a:g}, {b:g}, {g:g}", ""),
        ("Evidence overlay",
         f"{n_ev} item(s), {'on' if ov_on else 'off'}", "ev"),
    ])


# ----------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------

def overview(jurisdiction="National"):
    lam, a, b, g, mode = _params()
    _strip(jurisdiction)
    st.title("National Prioritization Overview")
    st.caption("Probability weighted across the historically grounded "
               "scenario set, integrated over eight domain agents. Every "
               "number traces to its source, calculation, and audit trail.")
    with st.spinner("Computing the integrated picture (first load only)"):
        M, res, _ = sc.cross_agent_run(jurisdiction, lam, a, b, g, mode, False)
        ind1 = sc.indicators("A1", jurisdiction, lam, a, b, g, mode)
        quad1 = sc.quadrants(ind1, st.session_state.get("g_theta", 0.7))
        prob = se.probability_table(lam, a, b, g)
        stab1 = se.stability_table("A1", jurisdiction, lam, a, b, g)

    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Final Integrated Stratification")
        qmap = dict(zip(quad1["Disease"], quad1["Quadrant"]))
        stabmap = stab1.set_index("Disease")
        for rank, d in enumerate(res.order, 1):
            t = res.tiers[d]
            cls = stabmap.loc[d, "Classification"] if d in stabmap.index else ""
            expo = stabmap.loc[d, "Exposure (sum p_s)"] if d in stabmap.index else 0
            why = (f"Overall rank {rank}; S = {res.scores[d]:.3f}; "
                   f"Agent 1 view: {cls.lower()}"
                   + (f", exposure {expo:g}" if cls == "Scenario sensitive" else ""))
            ui.tier_card(t, ui.SHORT.get(d, d), [qmap.get(d, "")], why,
                         is_target=(res.targets[t] == d))
    with right:
        st.subheader("Priority versus Stability")
        st.plotly_chart(ui.quadrant_chart(
            quad1, st.session_state.get("g_theta", 0.7),
            "Agent 1: Epidemiological Burden"), use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Scenario Probability Mass")
        st.plotly_chart(ui.prob_bars(prob), use_container_width=True)
    with c2:
        st.subheader("Active Signals")
        reg = load_scenario_registry()
        s7 = reg[reg["Scenario ID"] == "S7"]
        if not s7.empty:
            ui.signal("<b>S7 marker context.</b> " +
                      str(s7.iloc[0]["Sources"])[:160] + ".", warn=True)
        led = load_ledger()
        if not led.empty:
            v = int((led["status"] == "validated").sum())
            q = int((led["status"] == "quarantined").sum())
            ui.signal(f"<b>Evidence Gate.</b> {v} validated and applied; "
                      f"{q} quarantined with reasons; full ledger under "
                      "Evidence &amp; Data.")
        defaults = sum(1 for k in AGENTS
                       if wm.get_weights(k, AGENTS[k]["criteria"])[1] == "default")
        if defaults:
            ui.signal(f"<b>Elicitation.</b> Weights are defaults on "
                      f"{defaults} of 8 agents; supply expert judgment in "
                      "the Elicitation Studio.")
        st.subheader("How to read this")
        st.caption("Tier 1: inside the tolerance band of the best weighted "
                   "attainment; the target state is the member closest to "
                   "the ideal. Quadrants cross expected priority with "
                   "stability across scenarios. The overlay toggle and all "
                   "parameters live on each analysis page.")


# ----------------------------------------------------------------------
# Sensitivity & Simulation
# ----------------------------------------------------------------------

def sensitivity_page(jurisdiction="National"):
    lam, a, b, g, mode = _params()
    _strip(jurisdiction)
    st.title("Sensitivity & Simulation")
    st.caption("How much do conclusions depend on the parameters? Sweeps "
               "recompute the full chain; the Monte Carlo perturbs judgment "
               "inputs and reports robustness as probabilities.")

    st.subheader("1) Parameter sweeps (one at a time)")
    param = st.selectbox("Parameter", ["lam", "a", "b", "g"],
                         format_func={"lam": "lambda (expert share)",
                                      "a": "alpha (frequency exponent)",
                                      "b": "beta (severity exponent)",
                                      "g": "gamma (impact exponent)"}.get)
    grid = tuple(np.round(np.arange(0.0, 2.01, 0.25), 2)) if param != "lam" \
        else tuple(np.round(np.arange(0.0, 1.01, 0.1), 2))
    with st.spinner("Sweeping"):
        sw = sens.parameter_sweep(param, grid, jurisdiction, lam, a, b, g, mode)
    fig = go.Figure()
    for d in CORE4:
        dd = sw[sw["Disease"] == d]
        fig.add_trace(go.Scatter(x=dd["value"], y=dd["Final rank"],
                                 mode="lines+markers",
                                 name=ui.SHORT.get(d, d)))
    fig.update_yaxes(title="Final integrated rank", autorange="reversed",
                     dtick=1)
    fig.update_xaxes(title=param)
    st.plotly_chart(ui._base_layout(fig), use_container_width=True)
    st.caption("Flat lines mean the conclusion does not depend on the "
               "parameter; crossings mark the values where the "
               "prioritization would change.")

    st.subheader("2) Monte Carlo robustness of agent importance weights")
    n = st.slider("Draws", 100, 2000, 500, 100)
    conc = st.slider("Concentration (higher = smaller perturbations)",
                     10, 200, 60, 10)
    with st.spinner("Simulating"):
        tiers, ranks = sens.weight_monte_carlo(jurisdiction, lam, a, b, g,
                                               mode, n, float(conc))
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(tiers, width="stretch", hide_index=True)
        st.caption("Probability of each tier under Dirichlet perturbation "
                   "of W_g centered on the active weights.")
    with c2:
        fig = go.Figure()
        for d in CORE4:
            fig.add_trace(go.Box(
                y=ranks[ranks["Disease"] == d]["Rank"],
                name=ui.SHORT.get(d, d), boxmean=True))
        fig.update_yaxes(title="Integrated rank", autorange="reversed", dtick=1)
        st.plotly_chart(ui._base_layout(fig), use_container_width=True)

    st.subheader("3) Stability threshold and attainment mode")
    c3, c4 = st.columns(2)
    with c3:
        ak = st.selectbox("Agent for the theta sweep", list(AGENTS),
                          format_func=lambda k: AGENTS[k]["title"])
        ts = sens.theta_sweep(ak, jurisdiction, lam, a, b, g, mode)
        piv = ts.pivot_table(index="Disease", columns="theta",
                             values="Quadrant", aggfunc="first")
        piv.index = [ui.SHORT.get(i, i) for i in piv.index]
        st.dataframe(piv, width="stretch")
        st.caption("Quadrant membership as the stability cut moves; cells "
                   "that never change are theta robust conclusions.")
    with c4:
        mc = sens.mode_comparison(jurisdiction, lam, a, b, g)
        mc.index = [ui.SHORT.get(i, i) for i in mc.index]
        st.dataframe(mc, width="stretch")
        st.caption("The integrated conclusion under the three attainment "
                   "modes; the binary column is the first generation "
                   "ablation.")


# ----------------------------------------------------------------------
# Evidence & Data
# ----------------------------------------------------------------------

def evidence_data(jurisdiction="National"):
    _strip(jurisdiction)
    st.title("Evidence & Data")
    tabs = st.tabs(["Evidence Ledger", "Citations", "Derivations",
                    "Scenario Registry", "Data Dictionary"])
    with tabs[0]:
        led = load_ledger()
        if led.empty:
            st.info("No gate runs recorded yet; retrieve structured "
                    "evidence on any agent page.")
        else:
            f = st.multiselect("Status", sorted(led["status"].unique()),
                               default=list(sorted(led["status"].unique())))
            st.dataframe(led[led["status"].isin(f)], width="stretch",
                         hide_index=True)
            st.download_button("Download ledger (CSV)",
                               led.to_csv(index=False), "evidence_ledger.csv")
    with tabs[1]:
        st.dataframe(load_citations(), width="stretch", hide_index=True)
    with tabs[2]:
        st.dataframe(load_derivations(), width="stretch", hide_index=True)
        st.caption("The detailed calculation behind every derived baseline "
                   "figure: formula, numerator with source, denominator "
                   "with Census vintage, result, uncertainty.")
    with tabs[3]:
        st.dataframe(load_scenario_registry(), width="stretch",
                     hide_index=True)
    with tabs[4]:
        st.dataframe(load_data_dictionary(), width="stretch",
                     hide_index=True)


# ----------------------------------------------------------------------
# Elicitation Studio
# ----------------------------------------------------------------------

def elicitation_studio(jurisdiction="National"):
    _strip(jurisdiction)
    st.title("Elicitation Studio")
    st.caption("Every judgment input in one place: subcriterion weights and "
               "acceptable limits per agent, scenario intensities, and the "
               "agent importance weights, each through the four input modes "
               "with provenance.")

    rows = []
    for k in AGENTS:
        _, wprov = wm.get_weights(k, AGENTS[k]["criteria"])
        _, lprov = wm.get_expert_limits(k)
        rows.append({"Agent": AGENTS[k]["title"], "Weights": wprov,
                     "Limits": lprov})
    _, agprov = sc.get_agent_weights()
    st.subheader("Readiness")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(f"Agent importance weights W_g: {agprov}. Defaults are "
               "legitimate but deliberate choices; anything elicited below "
               "recomputes everywhere instantly.")

    st.subheader("Per agent elicitation")
    ak = st.selectbox("Agent", list(AGENTS),
                      format_func=lambda k: AGENTS[k]["title"],
                      key="studio_agent")
    with st.spinner("Loading the agent's evidence"):
        profile, result, *_ = run_agent(ak, jurisdiction, CORE4)
    t1, t2 = st.tabs(["Subcriterion weights", "Acceptable limits"])
    with t1:
        wm.render_weight_elicitation(ak, AGENTS[ak]["title"],
                                     AGENTS[ak]["criteria"])
    with t2:
        from modules.agent_engine import median_limits_for
        wm.render_limits_editor(ak, AGENTS[ak]["title"], profile,
                                AGENTS[ak]["criteria"],
                                median_limits_for(profile,
                                                  AGENTS[ak]["criteria"]))

    st.subheader("Agent importance weights (W_g)")
    wm.render_weight_elicitation("agent_level::global",
                                 "Cross agent integration", list(AGENTS))
    st.caption("Scenario severity and system impact are elicited on the "
               "Scenarios & Probability page beside their audit table.")
