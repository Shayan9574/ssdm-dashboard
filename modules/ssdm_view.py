"""
ssdm_view.py

SSDM Integration page: indicators and typology per agent, the perspective
weights W_g, the second order matrices (expected and baseline), the final
stratifications side by side, an AI written synthesis that never alters a
number, and the downloads with the run manifest.
"""

import json

import streamlit as st

from modules import state, controls, genai, ui
from modules import ssdm_core as sc
from modules import scenario_engine as se
from modules import weight_manager as wm
from modules.agent_engine import AGENTS


def run_manifest(jurisdiction: str) -> dict:
    from modules import feeds, live_subcriteria as ls
    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=str(feeds.ROOT)).stdout.strip()
    except Exception:
        commit = ""
    ref = ls.reference_week(jurisdiction)
    return {"jurisdiction": jurisdiction, "reference_week": str(ref)[:10] if ref is not None else "",
            "settings": state.settings(), "context_key": state.context_key(),
            "code_commit": commit or "unknown", "feeds": feeds.data_status().to_dict("records"),
            "inputs_store": state.backend().name}


def render(jurisdiction: str = "National") -> None:
    st.header("SSDM Integration & Outputs")
    controls.param_sliders()
    lam, alpha, beta, gamma = controls.params()
    theta = float(state.setting("theta_si"))

    st.subheader("1) Probability weighted indicators and typology")
    agent_key = st.selectbox("Agent", list(AGENTS), format_func=lambda k: AGENTS[k]["title"],
                             key="ss_agent")
    ind = sc.indicators(agent_key, jurisdiction, lam, alpha, beta, gamma)
    quad = sc.quadrants(ind, theta)
    st.dataframe(quad.round(4), width="stretch", hide_index=True)
    st.plotly_chart(ui.quadrant_chart(quad, theta, AGENTS[agent_key]["title"]), width="stretch")

    st.subheader("2) Perspective weights W_g")
    with st.expander("Elicit perspective weights"):
        wm.render_weight_elicitation("agent_level::global", "Cross agent integration", list(AGENTS))
    w, wprov = sc.get_agent_weights()
    st.caption(f"Active W_g ({wprov}): " + ", ".join(f"{sc.AGENT_LABELS[k]} = {v:.3f}" for k, v in w.items()))

    st.subheader("3) Second order matrices and final stratifications")
    M_exp, res_exp, _ = sc.cross_agent_run(jurisdiction, lam, alpha, beta, gamma, "", False)
    M_base, res_base, _ = sc.cross_agent_run(jurisdiction, lam, alpha, beta, gamma, "", True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Expected rank matrix (scenario weighted)**")
        st.dataframe(M_exp.round(4), width="stretch")
        st.markdown("**Final stratification (expected)**")
        st.dataframe(res_exp.table(), width="stretch", hide_index=True)
    with c2:
        st.markdown("**Baseline rank matrix (present conditions)**")
        st.dataframe(M_base.round(4), width="stretch")
        st.markdown("**Final stratification (baseline)**")
        st.dataframe(res_base.table(), width="stretch", hide_index=True)
    moved = [d for d in res_exp.order if res_base.tiers.get(d) != res_exp.tiers.get(d)]
    st.caption("Distinctions due to the scenario layer: " +
               (", ".join(f"{d} (tier {res_base.tiers.get(d)} at baseline, {res_exp.tiers.get(d)} expected)"
                          for d in moved) if moved else "none; both matrices give the same tiers.")
               + " Attainment margins (expected): " +
               ", ".join(f"tier {t}: {g:.4f}" for t, g in res_exp.separations.items()))

    st.subheader("4) Synthesis for decision makers (AI written, numbers unchanged)")
    if genai.gemini_available() and state.can_edit():
        if st.button("Write the synthesis"):
            quads = {AGENTS[k]["title"]: sc.quadrants(
                sc.indicators(k, jurisdiction, lam, alpha, beta, gamma), theta)[["Disease", "Quadrant"]]
                for k in AGENTS}
            prob = se.probability_table(lam, alpha, beta, gamma)[["Scenario ID", "Name", "p_s"]].round(4)
            from modules import evidence_gate as eg
            ext = eg.external_evidence()
            ext = ext[["disease", "metric", "value", "unit", "source_title"]].head(40) if not ext.empty else ext
            try:
                st.session_state["ss_synth"] = sc.synthesize(res_exp.table(), quads, prob, ext,
                                                             state.setting("gemini_model"))
            except Exception as e:  # noqa: BLE001
                st.error(f"Synthesis failed: {e}")
    if st.session_state.get("ss_synth"):
        st.markdown(st.session_state["ss_synth"])
        st.caption("Written by the AI assistant from the computed tables; every number comes from the engine.")

    st.subheader("5) Downloads")
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Final stratification (CSV)", res_exp.table().to_csv(index=False),
                       "ssdm_final_stratification.csv")
    d2.download_button("Second order matrix (CSV)", M_exp.to_csv(), "ssdm_second_order_matrix.csv")
    d3.download_button("Indicators of this agent (CSV)", quad.to_csv(index=False), "ssdm_indicators.csv")
    d4.download_button("Run manifest (JSON)", json.dumps(run_manifest(jurisdiction), indent=1, default=str),
                       "ssdm_run_manifest.json")
