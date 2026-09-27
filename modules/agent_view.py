"""
agent_view.py

The single renderer behind all eight agent pages. Shows, in order:
the agent header with its optimization space and engine provenance
badge; the baseline evidence matrix with the calculation audit
(Derivations sheet) beside it; the agent's live surveillance charts;
the elicitation panel (weights, expert limits) with the readiness
prompt; and the updated MOSDM stratification with its full audit
trail. Implements the standing rules: every derived number shows its
detailed calculation separately, results carry an engine provenance
label, and a co-equal tier triggers an in-dashboard prompt to elicit
expert limits and weights rather than a silent default.
"""

from typing import List
import pandas as pd
import streamlit as st

from modules import weight_manager as wm
from modules.agent_engine import AGENTS, CORE4, run_agent, median_limits_for
from modules.load_data import load_citations, load_derivations

ENGINE_BADGE = "Engine: second generation MOSDM (graded attainment, tolerance tiers, dominance guard)"
MODES = {"graded_calibrated": "Graded, discrimination calibrated anchors (default)",
         "graded_fixed": "Graded, fixed anchor theta = 0.5",
         "binary": "Binary attainment (first generation, ablation)"}


def render_agent(agent_key: str, jurisdiction: str = "National") -> None:
    cfg = AGENTS[agent_key]
    jur_label = "US National" if jurisdiction == "National" else "Michigan (MI)"

    st.header(cfg["title"], help=cfg["space"])
    st.caption(f"{cfg['space']}  •  Scope: **{jur_label}**")
    st.markdown(f":green-badge[{ENGINE_BADGE}]" if hasattr(st, "badge")
                else f"`{ENGINE_BADGE}`")

    selected = st.multiselect(
        "Diseases to evaluate", CORE4, default=CORE4,
        key=f"{agent_key}_sel",
        help="The four Phase 2 target pathogens; deselect to explore subsets.")
    if not selected:
        st.warning("Select at least one disease.")
        return

    profile, result, weights, wprov, limits, lprov = run_agent(
        agent_key, jurisdiction, selected)

    from modules import scenario_engine as se
    scen_opts = ["Baseline"] + se.scenario_ids(include_stress=True)
    reg_names = se.get_registry().set_index("Scenario ID")["Name"].to_dict()
    chosen = st.selectbox(
        "Decision environment", scen_opts,
        format_func=lambda s: "Baseline (elicited or default weights)"
        if s == "Baseline" else f"{s}: {reg_names.get(s, s)}"
        + (" [stress test, unweighted]" if s.startswith("ST") else ""),
        key=f"{agent_key}_scen",
        help="A scenario replaces the weight profile and re runs the full "
             "stratification; the disease data never change.")
    if chosen != "Baseline":
        result, sprov = se.run_scenario(agent_key, chosen, jurisdiction)
        st.caption(f"Scenario weight profile: {sprov}. Stability against the "
                   "baseline is on the Scenarios & Probability page.")

    # 1) baseline matrix + calculation audit
    st.subheader("1) Baseline Evidence Matrix",
                 help="Crisp parsed values from the curated evidence base; ranges are stored as their arithmetic midpoints with the raw form retained.")
    st.dataframe(profile, width="stretch", hide_index=True)
    with st.expander("Calculation audit: how every derived figure was computed"):
        der = load_derivations(target_diseases=selected)
        if der.empty:
            st.info("No derived figures among the displayed values; all cells "
                    "are direct source extractions (see citations below).")
        else:
            st.dataframe(der, width="stretch", hide_index=True)
            st.caption("Each row states the formula, numerator with source, "
                       "population denominator with Census vintage, result, "
                       "and uncertainty interval where published.")

    # 2) live surveillance
    if cfg["charts"]:
        st.subheader(f"2) Real-Time Surveillance ({jur_label})",
                     help="Weekly CDC feeds with a one week completed reporting lag.")
        cols = st.columns(len(cfg["charts"]))
        for col, (title, fn, takes_jur) in zip(cols, cfg["charts"]):
            with col:
                st.markdown(f"**{title}**")
                try:
                    series = fn(jurisdiction=jurisdiction) if takes_jur else fn(jurisdiction)
                    if series is None or (hasattr(series, "empty") and series.empty):
                        st.info("No time series available.")
                    else:
                        st.line_chart(series)
                except Exception as e:
                    st.caption(f"Chart unavailable: {e}")

    # 3) elicitation and readiness
    st.subheader("3) Weights & Acceptable Limits",
                 help="Section 7: four input modes; expert limits override the median default and stay fixed across iterations.")
    ready = wprov != "default" or lprov != "default"
    if not ready:
        st.info("Readiness note: this agent currently runs on defaults "
                "(equal weights, median limits). Elicit expert judgment "
                "below, or keep defaults deliberately.")
    with st.expander("Elicit subcriterion weights (four input modes)",
                     expanded=False):
        wm.render_weight_elicitation(agent_key, cfg["title"], cfg["criteria"])
    with st.expander("Expert acceptable limits", expanded=False):
        med = median_limits_for(profile, cfg["criteria"])
        wm.render_limits_editor(agent_key, cfg["title"], profile,
                                cfg["criteria"], med)
    st.caption(f"Active configuration: weights = {wprov}; limits = {lprov}. "
               "Change either above and the stratification below recomputes.")

    # 4) updated MOSDM stratification
    st.selectbox("Attainment mode", list(MODES), format_func=MODES.get,
                 key="attainment_mode",
                 help="Graded attainment anchors credit at the acceptable "
                      "limit and lets magnitude matter; binary is the first "
                      "generation step rule, kept for comparison.")
    st.subheader("4) Updated MOSDM Stratification",
                 help="Weighted attainment k^w with tolerance based tiers, "
                      "dominance guard, target state by smallest weighted "
                      "distance to the ideal, and within tier ordering by S_i.")
    tbl = result.table()
    st.dataframe(tbl, width="stretch", hide_index=True)

    tier1 = [a for a, t in result.tiers.items() if t == 1]
    if len(tier1) == len(selected) and len(selected) > 1:
        st.warning(
            "All selected diseases share one co-equal tier: their evidence "
            "profiles trade off and defaults cannot separate them. This is "
            "a legitimate result, and it is also the designed moment for "
            "judgment: set expert acceptable limits or elicit weights in "
            "Section 3 to differentiate, or accept the co-equal tier with "
            "the within tier ordering by weighted distance.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Per iteration audit**")
        for rec in result.iterations:
            with st.expander(f"Iteration {rec.iteration}: tier "
                             f"{', '.join(rec.tier)}"):
                st.write("Pool:", ", ".join(rec.pool))
                lim = pd.DataFrame({
                    "Limit": {c: round(v, 4) for c, v in rec.limits.items()},
                    "Source": rec.limit_source})
                st.dataframe(lim, width="stretch")
                st.write("Weighted attainment k^w:",
                         {a: round(v, 3) for a, v in rec.k_weighted.items()},
                         f" tolerance δ = {rec.delta:.3f} | mode: {rec.attainment_mode}")
                if rec.thetas:
                    st.write("Calibrated anchors θ_j:",
                             {c: round(t, 2) for c, t in rec.thetas.items()})
                if rec.demotions:
                    st.write("Dominance guard demotions:", rec.demotions)
                if rec.noninformative:
                    st.write("Noninformative this iteration:",
                             ", ".join(rec.noninformative))
                st.write(f"Target state T0: {rec.target}; within tier S_i:",
                         {a: round(v, 4) for a, v in rec.scores.items()})
    with c2:
        st.markdown("**Explanations**")
        for a in result.order:
            st.markdown(f"- **{a}** — {result.explanations[a]}")
        if result.separations:
            st.markdown("**Tier separations (S gap):** " + ", ".join(
                f"tier {t} to {t+1}: {g}" for t, g in result.separations.items()))

    # 4b) AI evidence enrichment (Section 5): strictly additive context
    with st.expander("AI evidence enrichment (additive, never enters the matrix)"):
        import os as _os
        from modules import evidence as _ev
        from modules import genai as _genai
        q = st.text_input("Enrichment query",
                          f"{' and '.join(selected[:2])} {cfg['title'].split(':')[1]} latest CDC evidence",
                          key=f"enr_q_{agent_key}")
        serper_key = _os.environ.get("SERPER_API_KEY", "")
        if st.button("Search and summarize", key=f"enr_b_{agent_key}",
                     disabled=not serper_key,
                     help="Requires SERPER_API_KEY in Colab Secrets; results are "
                          "context for the analyst and never modify any value."):
            try:
                hits = _ev.serper_search(serper_key, q, num=5)
                for h in hits:
                    st.markdown(f"- [{h['title']}]({h['link']}): {h['snippet']}")
                if _genai.gemini_available() and hits:
                    joined = "\n".join(f"{h['title']}: {h['snippet']}" for h in hits)
                    summary = _genai.call_gemini(
                        _os.environ.get("GEMINI_API_KEY", ""),
                        "gemini-2.5-flash",
                        "Summarize these search snippets in three sentences of "
                        "public health context. Cite nothing not present.\n" + joined)
                    st.info(summary)
            except Exception as e:
                st.error(f"Enrichment failed: {e}")
        if not serper_key:
            st.caption("Add SERPER_API_KEY in Colab Secrets (and Cell 4) to enable.")

    # 5) evidence base
    st.divider()
    with st.expander("Evidence Base & Research Citations"):
        cits = load_citations(target_diseases=selected)
        if not cits.empty:
            st.dataframe(cits[["Target Disease", "Target Metric",
                               "Source Organization", "Article/Report Title",
                               "Publication/Update Date", "Original Live URL"]],
                         width="stretch", hide_index=True)
