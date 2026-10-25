"""
agent_view.py

The single renderer behind the eight agent pages: evidence matrix with the
calculation audit, live surveillance, weights and limits with provenance,
the stratification with its complete per iteration audit trail, and the
external evidence retrieved for the agent (information only).
"""

import json
import os
import re

import numpy as np
import pandas as pd
import streamlit as st

from modules import state, ui
from modules import weight_manager as wm
from modules.agent_engine import AGENTS, CORE4, run_agent, default_limits, active_criteria
from modules.load_data import load_citations, load_derivations
from modules.controls import MODE_LABELS


def _fmt(v, nd=4):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)


def render_agent(agent_key: str, jurisdiction: str = "National") -> None:
    cfg = AGENTS[agent_key]
    jur_label = "US National" if jurisdiction == "National" else "Michigan (MI)"
    s = state.settings()
    st.header(cfg["title"], help=cfg["space"])
    st.caption(f"{cfg['space']}  •  Scope: **{jur_label}**  •  Engine: extended stratification core, "
               f"{MODE_LABELS[s['attainment_mode']].lower()}")

    selected = st.multiselect("Diseases to evaluate", CORE4, default=CORE4, key=f"{agent_key}_sel")
    if not selected:
        st.warning("Select at least one disease.")
        return
    from modules import scenario_engine as se
    reg_names = se.get_registry().set_index("Scenario ID")["Name"].to_dict()
    chosen = st.selectbox(
        "Decision environment", ["Baseline"] + se.scenario_ids(include_stress=True),
        format_func=lambda x: "Baseline (present conditions)" if x == "Baseline"
        else f"{x}: {reg_names.get(x, x)}" + (" [stress test, unweighted]" if x.startswith("ST") else ""),
        key=f"{agent_key}_scen",
        help="A scenario replaces the weight profile and the limits it tightens and re runs the "
             "whole stratification; the evidence never changes.")
    scen = None if chosen == "Baseline" else chosen
    profile, result, weights, wprov, limits, lprov = run_agent(agent_key, jurisdiction, selected, scen)

    # 1) evidence
    st.subheader("1) Evidence Matrix",
                 help="Values entering the engine. Ranges are fuzzy intervals whose graded mean "
                      "integration representation (the midpoint) is used; an upper bound such as "
                      "'<1%' is the interval from zero to the bound.")
    st.dataframe(profile, width="stretch", hide_index=True)
    from modules import live_subcriteria as ls
    from modules.agents.agent7_outbreak import rt_current
    notes = pd.concat([ls.freshness_report(jurisdiction),
                       pd.DataFrame(rt_current(jurisdiction)[1])], ignore_index=True)
    if not notes.empty:
        with st.expander(f"Live values treated as missing ({len(notes)})"):
            st.dataframe(notes.astype(str), width="stretch", hide_index=True)
    with st.expander("Calculation audit: how every derived figure was computed"):
        der = load_derivations(target_diseases=selected)
        if der.empty:
            st.info("No derived figures among the displayed values.")
        else:
            st.dataframe(der.astype(str), width="stretch", hide_index=True)

    # 2) live surveillance
    if cfg["charts"]:
        st.subheader(f"2) Surveillance ({jur_label})")
        cols = st.columns(len(cfg["charts"]))
        for col, (title, fn) in zip(cols, cfg["charts"]):
            with col:
                st.markdown(f"**{title}**")
                try:
                    series = fn(jurisdiction)
                    if series is None or series.empty:
                        st.info("No series for this scope.")
                    else:
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        for c in series.columns:
                            fig.add_trace(go.Scatter(x=series.index, y=series[c], mode="lines", name=str(c)))
                        st.plotly_chart(ui._base_layout(fig, height=240), width="stretch")
                except Exception as e:  # noqa: BLE001
                    st.caption(f"Chart unavailable: {e}")

    # 3) weights and limits
    st.subheader("3) Weights & Acceptable Limits")
    st.caption(f"Active in this environment: weights = {wprov}; limits = {lprov}.")
    wdf = pd.DataFrame({"Weight": {c: _fmt(v) for c, v in weights.items()},
                        "Limit used (expert or scenario)": {c: _fmt(limits.get(c)) for c in weights}})
    st.dataframe(wdf, width="stretch")
    if scen is not None:
        em = se.emphasis_multipliers(scen)
        em = em[em["Agent"] == agent_key] if not em.empty else em
        if not em.empty:
            with st.expander("Emphasis multipliers of this scenario"):
                st.dataframe(em.round(4).astype(str), width="stretch", hide_index=True)
    with st.expander("Elicit baseline weights (four input modes)"):
        wm.render_weight_elicitation(agent_key, cfg["title"], active_criteria(profile, agent_key))
    with st.expander("Expert acceptable limits"):
        wm.render_limits_editor(agent_key, cfg["title"], cfg["criteria"],
                                default_limits(agent_key, jurisdiction, tuple(selected)))

    # 4) stratification
    st.subheader("4) Stratification",
                 help="Weighted attainment with tolerance tiers, the dominance guard, the target "
                      "closest to the ideal point, the within tier distance S_i, and the attainment "
                      "margin of every tier boundary.")
    st.dataframe(result.table(), width="stretch", hide_index=True)
    if result.unstratified:
        st.warning("Not stratifiable (observed on no active criterion): " + ", ".join(result.unstratified))
    if result.inactive_criteria:
        st.caption("Inactive under the coverage rule: " + ", ".join(result.inactive_criteria))
    tier1 = [a for a, t in result.tiers.items() if t == 1]
    if len(tier1) == len(result.order) and len(result.order) > 1:
        st.info("Every disease shares the first tier: the evidence trades off and the weights and "
                "limits in place cannot separate them. Expert limits or weights are the designed "
                "way to resolve this; the within tier order by S_i is shown.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Audit trail, iteration by iteration**")
        for rec in result.iterations:
            with st.expander(f"Iteration {rec.iteration}: tier {rec.iteration} = {', '.join(rec.tier)}"):
                st.write("Pool:", ", ".join(rec.pool))
                lim = pd.DataFrame({"Limit": {c: _fmt(v) for c, v in rec.limits.items()},
                                    "Source": rec.limit_source,
                                    "Limit position": {c: _fmt(v) for c, v in rec.limit_positions.items()},
                                    "Anchor theta": {c: _fmt(v, 2) for c, v in rec.thetas.items()}})
                st.dataframe(lim.astype(str), width="stretch")
                att = pd.DataFrame(rec.attainment).T.round(4)
                att["K (weighted attainment)"] = pd.Series(rec.k_weighted).round(4)
                st.dataframe(att.astype(str), width="stretch")
                st.write(f"Tolerance delta = {rec.delta:.4f}; candidate tier: {', '.join(rec.candidate_tier)}")
                if rec.demotions:
                    st.write("Dominance guard demotions: " +
                             "; ".join(f"{a} by {b}" for a, b in rec.demotions))
                if rec.noninformative:
                    st.write("Noninformative this iteration: " + ", ".join(rec.noninformative))
                st.write(f"Target T: {rec.target}; distance to ideal: "
                         + ", ".join(f"{a} {v:.4f}" for a, v in rec.ideal_distance.items() if v == v))
                st.write("Within tier S_i: " + ", ".join(f"{a} {v:.4f}" for a, v in rec.scores.items() if v == v))
    with c2:
        st.markdown("**Explanations**")
        for a in result.order + result.unstratified:
            st.markdown(f"- **{a}**: {result.explanations[a]}")
        if result.separations:
            st.markdown("**Attainment margins:** " + ", ".join(
                f"tier {t} to {t + 1}: {g:.4f}" for t, g in result.separations.items()))
            if any(g < 0 for g in result.separations.values()):
                st.caption("A negative margin marks a boundary imposed by the dominance guard.")

    # 5) external evidence (information only)
    _external_block(agent_key, cfg, profile, selected)

    st.divider()
    with st.expander("Evidence base and citations"):
        cits = load_citations(target_diseases=selected)
        if not cits.empty:
            keep = [c for c in ["Target Disease", "Target Metric", "Source Organization",
                                "Article/Report Title", "Publication/Update Date", "Original Live URL"]
                    if c in cits.columns]
            st.dataframe(cits[keep].astype(str), width="stretch", hide_index=True)


def _external_block(agent_key, cfg, profile, selected):
    from modules import evidence_gate as eg, genai
    st.subheader("5) External Evidence for Decision Makers",
                 help="Figures from official external sources, validated by four checks. They are "
                      "shown beside the dashboard's values and never change the stratification.")
    ext = eg.external_evidence(agent_key)
    if ext.empty:
        st.caption("No validated external evidence yet for this agent.")
    else:
        show = ext.copy()
        for c in ("value", "dashboard_value", "relative_difference"):
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors="coerce").round(4)
        st.dataframe(show.astype(str), width="stretch", hide_index=True)
        if "differs_notably" in ext.columns and ext["differs_notably"].astype(bool).any():
            st.caption("Rows flagged 'differs_notably' report a value more than 25 percent away from "
                       "the dashboard's value: a signal for curators to review the evidence base.")
    with st.expander("Retrieve external evidence (AI assistant, four check gate)"):
        if not state.can_edit():
            st.info("Sign in as editor to retrieve evidence.")
            return
        if not genai.gemini_available():
            st.info("No AI key configured.")
            return
        q = st.text_input("Scope", f"{', '.join(selected)}: {cfg['title'].split(':')[1].strip()}",
                          key=f"enr_q_{agent_key}")
        if st.button("Retrieve and check", key=f"enr_b_{agent_key}"):
            prompt = (
                "You are the external evidence tier of a public health prioritization system. "
                "Using live web search, gather the most decision relevant, recent, quantitative "
                f"evidence for EVERY disease listed, within this domain: {q}.\n"
                f"Existing subcriteria in this domain: {cfg['criteria']}.\n"
                "Return ONLY a JSON array (no markdown fences). Each element: "
                '{"disease": str, "metric": str, "value": str, "unit": str, "timeframe": str, '
                '"source_title": str, "source_url": str, "maps_to": "<one existing subcriterion, '
                'or NEW: <proposed name>>", "use": "one sentence on how a decision maker can use '
                'this number"}. Report each value in the unit the source uses. Include at least '
                "one row per disease; prefer official sources (CDC, WHO, peer reviewed); include "
                "new indicators and policy or advisory signals where relevant.")
            try:
                model = state.setting("gemini_model")
                text, _ = genai._with_fallback(genai.call_gemini_grounded,
                                               os.environ.get("GEMINI_API_KEY", ""), model, prompt)
                raw = re.sub(r"```(json)?", "", text).strip()
                data = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
                with st.spinner("Checking source, schema, definition, corroboration"):
                    report = eg.run_gate(data, profile, cfg["criteria"], agent_key, model)
                eg.append_ledger(report)
                st.session_state[f"enr_rep_{agent_key}"] = report
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Retrieval failed: {e}")
        rep = st.session_state.get(f"enr_rep_{agent_key}")
        if rep is not None:
            st.markdown("**Latest gate report (every row, every check, every reason)**")
            st.dataframe(rep.astype(str), width="stretch", hide_index=True)
