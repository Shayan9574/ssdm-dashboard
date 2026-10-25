"""
scenario_view.py

Scenarios & Probability page: the registry with the weekly recomputed
admission (episodes, windows, frequencies beside the curated counts), the
data derived emphasis multipliers and limit overrides, the severity and
impact judgments (expert respondents and the auxiliary AI assessor), the
scenario weight audit table, and the re runs with stability.
"""

import pandas as pd
import streamlit as st

from modules import state, controls, genai, ui
from modules import scenario_engine as se
from modules.weight_manager import INTENSITY_SCALE
from modules.agent_engine import AGENTS


def render(jurisdiction: str = "National") -> None:
    st.header("Scenarios & Probability")
    st.caption("A scenario changes the decision environment (weights and the limits it tightens), "
               "never the evidence; every agent is re stratified under every scenario.")
    reg = se.get_registry()

    st.subheader("1) Registry and weekly admission")
    adm = se.admission_table()
    show = adm.copy()
    show["f_s 95% interval"] = show["f_s 95% interval"].apply(
        lambda t: "" if t[0] != t[0] else f"{t[0]:.3f} to {t[1]:.3f}")
    for c in ("Y_s", "f_s"):
        show[c] = show[c].round(3)
    st.dataframe(show.drop(columns=["weighted"]).astype(str), width="stretch", hide_index=True)
    st.caption(f"Episode rule: marker weeks joined across interruptions of at most "
               f"{state.setting('episode_gap_weeks')} weeks; an episode holds the marker in at least "
               f"{state.setting('episode_min_weeks')} weeks (S4: three consecutive weeks by definition). "
               "The interval is the exact Poisson 95% interval of the episode rate. ST1 has no countable "
               "marker and runs as an unweighted stress test.")
    with st.expander("Marker conditions and sources"):
        st.dataframe(reg[["Scenario ID", "Name", "Status", "Marker Condition", "Episodes", "Sources"]].astype(str),
                     width="stretch", hide_index=True)

    st.subheader("2) Emphasis and limit overrides")
    sid = st.selectbox("Scenario", se.scenario_ids(True), key="scen_em_pick")
    em = se.emphasis_multipliers(sid)
    if em.empty:
        st.caption("No emphasized criteria.")
    else:
        st.dataframe(em.round(4).astype(str), width="stretch", hide_index=True)
    ov = {k: se.scenario_limit_overrides(sid, k, jurisdiction) for k in AGENTS}
    ov = {k: v for k, v in ov.items() if v}
    if ov:
        st.dataframe(pd.DataFrame([{"Agent": k, "Criterion": c, "Limit under the scenario": round(x, 4)}
                                   for k, v in ov.items() for c, x in v.items()]),
                     width="stretch", hide_index=True)
        st.caption(f"Tightening: emphasized criteria with a history take its "
                   f"P{int(round(100 * state.setting('tightening_quantile')))} inside the limits window; "
                   "Rt takes its marker threshold 1.")

    st.subheader("3) Severity and System Impact")
    controls.param_sliders(("lambda", "alpha", "beta", "gamma"))
    lam, alpha, beta, gamma = controls.params()
    if state.can_edit():
        with st.expander("Record expert ratings (one respondent at a time)"):
            c1, c2, c3, c4 = st.columns(4)
            s_pick = c1.selectbox("Scenario", se.scenario_ids(True), key="exp_sid")
            who = c2.text_input("Respondent", key="exp_who")
            sev = c3.selectbox("Severity", list(INTENSITY_SCALE), key="exp_sev")
            imp = c4.selectbox("System impact", list(INTENSITY_SCALE), key="exp_imp")
            b1, b2 = st.columns(2)
            if b1.button("Save rating", disabled=not who):
                se.set_expert_terms(s_pick, sev, imp, who)
                st.rerun()
            if b2.button("Remove this respondent's rating", disabled=not who):
                se.remove_expert(s_pick, who)
                st.rerun()
        if genai.gemini_available() and st.button("Assess every scenario with the AI assessor"):
            prog = st.progress(0.0, "Assessing")
            rows = reg.to_dict("records")
            for i, r in enumerate(rows):
                try:
                    sv, iv, just = se.gemini_assess(pd.Series(r), model=state.setting("gemini_model"))
                    se.set_ai_assessment(r["Scenario ID"], sv, iv, just, state.setting("gemini_model"))
                except Exception as e:  # noqa: BLE001
                    st.warning(f"{r['Scenario ID']}: assessment failed ({e})")
                prog.progress((i + 1) / len(rows))
            st.rerun()
    rows = []
    for s_id in se.scenario_ids(True):
        a = se.get_assessments(s_id)
        for who_, t in a.get("expert_respondents", {}).items():
            rows.append({"Scenario": s_id, "Assessor": f"expert: {who_}", "Severity": t["severity"],
                         "Impact": t["impact"], "Note": ""})
        if a.get("ai"):
            rows.append({"Scenario": s_id, "Assessor": f"AI ({a['ai'].get('model', '')})",
                         "Severity": a["ai"]["severity"], "Impact": a["ai"]["impact"],
                         "Note": a["ai"].get("justification", "")})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.caption("No ratings yet: every scenario uses the neutral Moderate term, flagged as default.")

    st.subheader("4) Scenario weights")
    prob = se.probability_table(lam, alpha, beta, gamma)
    disp = prob.copy()
    for c in ("Y_s", "f_s", "v_s", "h_s", "pi_s", "p_s"):
        disp[c] = disp[c].round(4)
    st.dataframe(disp.drop(columns=["weighted"]).astype(str), width="stretch", hide_index=True)
    chart = prob.dropna(subset=["p_s"])
    if not chart.empty:
        st.plotly_chart(ui.prob_bars(chart), width="stretch")
    st.caption("pi_s = f_s^alpha v_s^beta h_s^gamma, normalized over the weighted scenarios; "
               "the baseline is the reference outside the normalization. The weights are relative "
               "importances of conditions that can overlap, not probabilities of disjoint events.")

    st.subheader("5) Re runs and stability")
    agent_key = st.selectbox("Agent", list(AGENTS), format_func=lambda k: AGENTS[k]["title"],
                             key="scen_agent")
    with st.spinner("Re stratifying under every scenario"):
        tm = se.tier_matrix(agent_key, jurisdiction)
        stab = se.stability_table(agent_key, jurisdiction, lam, alpha, beta, gamma)
    st.markdown("**Tier matrix (1 = top tier)**")
    st.dataframe(tm, width="stretch")
    st.markdown("**Stability against the baseline**")
    sd = stab.copy()
    for c in ("Exposure E", "Instability I"):
        sd[c] = pd.to_numeric(sd[c], errors="coerce").round(4)
    st.dataframe(sd.astype(str), width="stretch", hide_index=True)
    if genai.gemini_available() and state.can_edit() and st.button("AI reading of the tier matrix"):
        from modules import ssdm_core as sc
        try:
            st.info(sc.interpret_tiers(AGENTS[agent_key]["title"], tm.reset_index(),
                                       state.setting("gemini_model")))
        except Exception as e:  # noqa: BLE001
            st.error(f"Interpretation failed: {e}")
