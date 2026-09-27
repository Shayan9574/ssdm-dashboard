"""
ssdm_view.py

The SSDM Integration & Outputs page (Sections 14 to 19): per agent
indicators and quadrant classification, the second order matrix with
agent level weight elicitation, the cross agent second generation
MOSDM on expected and baseline matrices in parallel, the AI synthesis
(clearly labeled, never altering results), and downloads.
"""

import pandas as pd
import streamlit as st

from modules import ssdm_core as sc
from modules import scenario_engine as se
from modules import weight_manager as wm
from modules.agent_engine import AGENTS
from modules import genai


def render(jurisdiction: str = "National") -> None:
    st.header("SSDM Integration & Outputs",
              help="Sections 14 to 19: probability weighted indicators, "
                   "quadrant classes, the second order matrix, the cross "
                   "agent run, and the final synthesis.")
    mode = st.session_state.get("attainment_mode", "graded_calibrated")
    c1, c2, c3, c4, c5 = st.columns(5)
    lam = c1.slider("lambda", 0.0, 1.0, 0.7, 0.05, key="ss_lam")
    alpha = c2.slider("alpha", 0.0, 2.0, 1.0, 0.1, key="ss_a")
    beta = c3.slider("beta", 0.0, 2.0, 1.0, 0.1, key="ss_b")
    gamma = c4.slider("gamma", 0.0, 2.0, 1.0, 0.1, key="ss_g")
    theta = c5.slider("theta (stability cut)", 0.5, 0.9, 0.7, 0.05,
                      help="High stability when SI is at least theta.")

    # 1) indicators and quadrants per agent
    st.subheader("1) Probability Weighted Indicators and Quadrants",
                 help="R, Phi, sigma, SI over the weighted scenarios; the "
                      "baseline stays outside as the reference.")
    agent_key = st.selectbox("Agent", list(AGENTS),
                             format_func=lambda k: AGENTS[k]["title"],
                             key="ss_agent")
    with st.spinner("Aggregating over scenarios"):
        ind = sc.indicators(agent_key, jurisdiction, lam, alpha, beta, gamma, mode)
        quad = sc.quadrants(ind, theta)
    st.dataframe(quad, width="stretch", hide_index=True)
    plot = quad.rename(columns={"R (expected rank)": "R", "SI (stability)": "SI"})
    st.scatter_chart(plot, x="R", y="SI", color="Quadrant", size=60)
    st.caption("Left of the rank midline: high priority. Above theta: high "
               "stability. Structural priority (high, stable), conditionally "
               "critical (high, unstable), stable low priority, latent risk.")

    # 2) agent level weights
    st.subheader("2) Agent Importance Weights (W_g)",
                 help="Section 7.2 agent level mode; equal weights default, "
                      "elicitable through the same four input modes.")
    with st.expander("Elicit agent level weights"):
        wm.render_weight_elicitation("agent_level::global",
                                     "Cross agent integration", list(AGENTS))
    w, wprov = sc.get_agent_weights()
    st.caption("Active W_g (" + wprov + "): " +
               ", ".join(f"{sc.AGENT_LABELS[k]} = {v:.3f}" for k, v in w.items()))

    # 3) second order matrix and cross agent run
    st.subheader("3) Second Order Matrix and Cross Agent SSDM",
                 help="Entries are expected ranks R_i per agent (Agent 5 "
                      "inverted at integration so lower always means greater "
                      "concern); the second generation MOSDM runs on this "
                      "matrix with direction negative.")
    with st.spinner("Running the cross agent integration"):
        M_exp, res_exp, _ = sc.cross_agent_run(jurisdiction, lam, alpha,
                                               beta, gamma, mode, False)
        M_base, res_base, _ = sc.cross_agent_run(jurisdiction, lam, alpha,
                                                 beta, gamma, mode, True)
    st.markdown("**Expected rank matrix (probability weighted)**")
    st.dataframe(M_exp.round(3), width="stretch")
    st.markdown("**Final integrated stratification (expected)**")
    st.dataframe(res_exp.table(), width="stretch", hide_index=True)
    with st.expander("Parallel run on the baseline rank matrix (reference)"):
        st.dataframe(M_base.round(3), width="stretch")
        st.dataframe(res_base.table(), width="stretch", hide_index=True)
        st.caption("Agreement between the two runs indicates the integrated "
                   "prioritization is not an artifact of the scenario set.")

    # 4) synthesis
    st.subheader("4) Final Synthesis (AI written, results unchanged)")
    model = st.text_input("Gemini model", se.DEFAULT_GEMINI_MODEL, key="ss_model")
    if st.button("Generate synthesis", disabled=not genai.gemini_available()):
        with st.spinner("Writing the synthesis"):
            quads = {k: sc.quadrants(
                sc.indicators(k, jurisdiction, lam, alpha, beta, gamma, mode),
                theta)[["Disease", "Quadrant"]] for k in AGENTS}
            prob = se.probability_table(lam, alpha, beta, gamma)[
                ["Scenario ID", "Name", "p_s"]]
            try:
                text = sc.synthesize(res_exp.table(), quads, prob, model)
                st.session_state["ss_synth"] = text
            except Exception as e:
                st.error(f"Synthesis failed: {e}")
    if st.session_state.get("ss_synth"):
        st.markdown(st.session_state["ss_synth"])
        st.caption("Narrative generated by the AI tier from the computed "
                   "tables; every number originates in the engine above.")

    # 5) outputs
    st.subheader("5) Downloads")
    st.download_button("Final stratification (CSV)",
                       res_exp.table().to_csv(index=False),
                       "ssdm_final_stratification.csv")
    st.download_button("Second order matrix (CSV)",
                       M_exp.to_csv(), "ssdm_second_order_matrix.csv")
    st.download_button("Indicators, selected agent (CSV)",
                       quad.to_csv(index=False), "ssdm_indicators.csv")
