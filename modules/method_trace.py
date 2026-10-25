"""
method_trace.py

Method Trace page: the complete chain of the method for the current run,
step by step in the order of the paper's Methodology section, with the
numbers each step produced. It is the reproducible record a reader of the
paper can follow from the evidence to the final stratification.
"""

import pandas as pd
import streamlit as st

from modules import state, controls
from modules import scenario_engine as se
from modules import ssdm_core as sc
from modules import live_subcriteria as ls
from modules import feeds
from modules.agent_engine import AGENTS, CORE4, run_agent, active_criteria


def render(jurisdiction: str = "National") -> None:
    st.title("Method Trace")
    st.caption("Every step of the method for the current run, with its numbers. Settings: "
               + ", ".join(f"{k} = {v}" for k, v in state.settings().items()))
    lam, alpha, beta, gamma = controls.params()
    theta = float(state.setting("theta_si"))

    st.subheader("Step 1. Evidence base and surveillance")
    st.dataframe(feeds.data_status(), width="stretch", hide_index=True)
    ref = ls.reference_week(jurisdiction)
    st.caption(f"Reference week: {str(ref)[:10] if ref is not None else 'none'}; live values older than "
               f"{ls.MAX_AGE_WEEKS} weeks are treated as missing.")

    st.subheader("Step 2. History window and historical limits")
    hs = ls.historical_stats(jurisdiction, 1, state.setting("history_window"))
    st.dataframe(pd.DataFrame(hs).T, width="stretch")

    st.subheader("Step 3. Decision matrices, coverage and weights per agent")
    ak = st.selectbox("Agent", list(AGENTS), format_func=lambda k: AGENTS[k]["title"], key="mt_agent")
    prof, res, w, wprov, lim, lprov = run_agent(ak, jurisdiction, CORE4)
    st.dataframe(prof, width="stretch", hide_index=True)
    st.caption(f"Active criteria: {', '.join(active_criteria(prof, ak))}. Weights ({wprov}): "
               + ", ".join(f"{c} = {v:.4f}" for c, v in w.items()))

    st.subheader("Step 4. Stratification core, iteration by iteration")
    for rec in res.iterations:
        st.markdown(f"**Iteration {rec.iteration}** (pool: {', '.join(rec.pool)})")
        t = pd.DataFrame({"Limit": rec.limits, "Source": rec.limit_source,
                          "Limit position": rec.limit_positions, "Anchor": rec.thetas})
        st.dataframe(t.astype(str), width="stretch")
        a = pd.DataFrame(rec.attainment).T
        a["K"] = pd.Series(rec.k_weighted)
        st.dataframe(a.round(4), width="stretch")
        st.caption(f"delta = {rec.delta:.4f}; candidate tier {rec.candidate_tier}; demotions "
                   f"{rec.demotions or 'none'}; tier {rec.tier}; target {rec.target}; "
                   f"S = {({k: round(v, 4) for k, v in rec.scores.items() if v == v})}")
    st.dataframe(res.table(), width="stretch", hide_index=True)

    st.subheader("Step 5. Scenario admission and frequency")
    adm = se.admission_table()
    st.dataframe(adm[["Scenario ID", "n_s", "Y_s", "f_s", "Source", "Episodes", "Observable window"]]
                 .round(4).astype(str), width="stretch", hide_index=True)

    st.subheader("Step 6. Scenario environments: emphasis and tightened limits")
    em = pd.concat([se.emphasis_multipliers(s) for s in se.scenario_ids(True)], ignore_index=True)
    st.dataframe(em.round(4).astype(str), width="stretch", hide_index=True)

    st.subheader("Step 7. Scenario weights")
    st.dataframe(se.probability_table(lam, alpha, beta, gamma).round(4).astype(str),
                 width="stretch", hide_index=True)

    st.subheader("Step 8. Re stratification and stability")
    st.dataframe(se.tier_matrix(ak, jurisdiction), width="stretch")
    st.dataframe(se.stability_table(ak, jurisdiction, lam, alpha, beta, gamma).round(4).astype(str),
                 width="stretch", hide_index=True)

    st.subheader("Step 9. Indicators and typology")
    st.dataframe(sc.quadrants(sc.indicators(ak, jurisdiction, lam, alpha, beta, gamma), theta).round(4),
                 width="stretch", hide_index=True)

    st.subheader("Step 10. Integration")
    M, r, wp = sc.cross_agent_run(jurisdiction, lam, alpha, beta, gamma, "", False)
    st.dataframe(M.round(4), width="stretch")
    st.caption(f"Perspective weights: {wp}")
    st.dataframe(r.table(), width="stretch", hide_index=True)
    for rec in r.iterations:
        st.caption(f"Integration iteration {rec.iteration}: K = "
                   f"{({k: round(v, 4) for k, v in rec.k_weighted.items()})}, delta = {rec.delta:.4f}, "
                   f"tier {rec.tier}, demotions {rec.demotions or 'none'}")
