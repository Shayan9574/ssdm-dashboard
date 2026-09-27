"""
scenario_view.py

The Scenarios & Probability page (Sections 10 to 13 in the interface):
the registry with markers, episodes and sources; severity and system
impact elicitation (expert terms and the Gemini assessor) with the
lambda blend and the exponents; the pi_s and p_s audit table; the per
agent tier matrix across all scenarios including the labeled stress
test; and the stability classification against the baseline.
"""

import os
import pandas as pd
import streamlit as st

from modules import scenario_engine as se
from modules.weight_manager import INTENSITY_SCALE, gmir
from modules.agent_engine import AGENTS
from modules import genai


def render(jurisdiction: str = "National") -> None:
    st.header("Scenarios & Probability",
              help="Sections 10 to 13: the historically constructed scenario "
                   "set, the three component probability rule, scenario wise "
                   "re stratification, and stability against the baseline.")
    st.caption("A scenario modifies the decision environment, never the "
               "disease data: its weight profile re runs the full second "
               "generation MOSDM inside every agent.")

    reg = se.get_registry()

    # 1) registry
    st.subheader("1) Scenario Registry (constructed from 2016 to 2026 system history)")
    show = reg[["Scenario ID", "Name", "Status", "Marker Condition",
                "Episodes", "n_s", "Y_s (years observable)", "Sources"]]
    st.dataframe(show, width="stretch", hide_index=True)
    st.caption("Admission required at least one verifiable historical "
               "occurrence; the ST1 economic lens has no countable marker and "
               "runs as a clearly labeled, unweighted stress test outside the "
               "probability normalization.")

    # 2) severity and system impact elicitation
    st.subheader("2) Severity and System Impact",
                 help="Each quantity elicited on the four term fuzzy scale "
                      "(GMIR defuzzification), from experts and from the "
                      "Gemini assessor, blended with lambda (expert dominant).")
    c1, c2, c3, c4, c5 = st.columns(5)
    lam = c1.slider("lambda (expert share)", 0.0, 1.0, 0.7, 0.05)
    alpha = c2.slider("alpha (frequency)", 0.0, 2.0, 1.0, 0.1)
    beta = c3.slider("beta (severity)", 0.0, 2.0, 1.0, 0.1)
    gamma = c4.slider("gamma (system impact)", 0.0, 2.0, 1.0, 0.1)
    model = c5.text_input("Gemini model", se.DEFAULT_GEMINI_MODEL)

    if st.button("Assess all scenarios with Gemini",
                 disabled=not genai.gemini_available(),
                 help="Sends each scenario's documented episodes and sources "
                      "to the model; add GEMINI_API_KEY in Colab Secrets to enable."):
        prog = st.progress(0.0, "Assessing")
        rows = reg.to_dict("records")
        for i, r in enumerate(rows):
            try:
                sev, imp, just = se.gemini_assess(pd.Series(r), model=model)
                se.set_ai_assessment(r["Scenario ID"], sev, imp, just, model)
            except Exception as e:
                st.warning(f"{r['Scenario ID']}: assessment failed ({e})")
            prog.progress((i + 1) / len(rows))
        prog.empty()
        st.success("Gemini assessments stored; every rating and its "
                   "justification appears in the expanders below.")

    with st.expander("Expert ratings and AI justifications, per scenario"):
        for _, r in reg.iterrows():
            sid = r["Scenario ID"]
            st.markdown(f"**{sid}: {r['Name']}**")
            e1, e2 = st.columns(2)
            a = se.get_assessments(sid)
            with e1:
                sev = st.selectbox(f"Expert severity ({sid})",
                                   ["(not rated)"] + list(INTENSITY_SCALE),
                                   key=f"sev_{sid}")
                imp = st.selectbox(f"Expert system impact ({sid})",
                                   ["(not rated)"] + list(INTENSITY_SCALE),
                                   key=f"imp_{sid}")
                if st.button(f"Apply expert rating ({sid})", key=f"exp_{sid}"):
                    if sev != "(not rated)" and imp != "(not rated)":
                        se.set_expert_terms(sid, sev, imp)
                        st.success(f"{sid}: severity {sev} "
                                   f"(GMIR {gmir(*INTENSITY_SCALE[sev]):.3f}), "
                                   f"impact {imp} "
                                   f"(GMIR {gmir(*INTENSITY_SCALE[imp]):.3f})")
            with e2:
                ai = a.get("ai")
                if ai:
                    st.write(f"Gemini ({ai['model']}): severity {ai['severity']}, "
                             f"impact {ai['impact']}")
                    st.caption(ai.get("justification", ""))
                else:
                    st.caption("No AI assessment yet.")
            st.divider()

    # 3) probability audit table
    st.subheader("3) Scenario Probabilities",
                 help="pi_s = f_s^alpha * s_s^beta * h_s^gamma, normalized "
                      "over the weighted scenarios; the baseline stays outside "
                      "the aggregation as the stability reference.")
    prob = se.probability_table(lam, alpha, beta, gamma)
    st.dataframe(prob, width="stretch", hide_index=True)
    chart = prob.dropna(subset=["p_s"]).set_index("Scenario ID")["p_s"]
    if not chart.empty:
        st.bar_chart(chart)

    # 4) re runs and stability per agent
    st.subheader("4) Scenario Re Runs and Stability",
                 help="Tier matrix: the full stratification re run under "
                      "every scenario's weight profile; stability judged "
                      "against the baseline, weighted by p_s.")
    agent_key = st.selectbox("Agent", list(AGENTS),
                             format_func=lambda k: AGENTS[k]["title"])
    with st.spinner("Re running the stratification under every scenario"):
        tm = se.tier_matrix(agent_key, jurisdiction)
        stab = se.stability_table(agent_key, jurisdiction,
                                  lam, alpha, beta, gamma)
    st.markdown("**Tier matrix (diseases by scenarios; 1 = top tier)**")
    st.dataframe(tm, width="stretch")
    st.markdown("**Stability classification (Section 13)**")
    st.dataframe(stab, width="stretch", hide_index=True)
    if st.button("AI tier interpretation", disabled=not genai.gemini_available(),
                 help="Three sentence reading of the matrix; results unchanged."):
        from modules import ssdm_core as sc
        try:
            st.info(sc.interpret_tiers(AGENTS[agent_key]["title"],
                                       tm.reset_index(), model))
        except Exception as e:
            st.error(f"Interpretation failed: {e}")
    st.caption("Robust priority: the tier holds across every weighted "
               "scenario. Scenario sensitive: displacement recorded with its "
               "magnitude, direction, triggering scenarios, exposure, and the "
               "probability weighted instability index. The stress test "
               "column is shown for policy insight and enters none of these "
               "quantities.")
