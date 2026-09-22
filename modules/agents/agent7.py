import streamlit as st
import pandas as pd
from modules.agent_m_data import get_active_decision_matrix
from modules.load_data import load_citations
from modules.live_connectors import get_rt_trends, get_wastewater_trends
from modules.scenarios import SCENARIOS_12
from modules.agents.agent7_outbreak import (
    AGENT_TITLE,
    build_agent7_profile,
    mosdm_tiers,
    weighted_tiebreak,
    scenario_weights_emphasis,
)

A7_KEY = "A7_"

def render(jurisdiction: str = "National"):
    jur_label = "US National" if jurisdiction == "National" else "Michigan (MI)"
    st.header(
        AGENT_TITLE,
        help="Monitors active epidemic velocity (Rt), outbreak growth probability (P > 0.5), and municipal wastewater viral concentrations (copies/L)."
    )
    st.caption(f"Real-time transmission velocity ($R_t$), outbreak growth probability, and sewershed viral load for **{jur_label}**.")

    with st.expander("💡 What You Are Looking At (Plain-English Guide)", expanded=False):
        st.markdown("""
        * **Effective Reproduction Number (Rt):** How fast the disease is actively spreading right now.
          * **$R_t > 1.0$ (Outbreak Expanding):** Each infected person spreads to >1 new person. Epidemic is growing.
          * **$R_t < 1.0$ (Outbreak Decelerating):** Spread is slowing down. Cases are declining.
        * **Probability of Outbreak Growth (P > 0.5):** Statistical confidence that current infections are multiplying.
        * **Wastewater Viral Concentration (copies/L):** Viral RNA in municipal wastewater. Detects surges **1 to 2 weeks before clinic visits**.
        * **Geographic Containment:** Whether the disease is widespread across all counties, regional, or localized.
        """)

    active_df = get_active_decision_matrix(jurisdiction=jurisdiction)
    
    all_diseases = active_df["Disease Type"].tolist()
    core_4 = ["Influenza", "COVID-19", "Respiratory Syncytial Virus (RSV)", "Meningococcal Disease (Meningitis)"]
    default_selection = [d for d in core_4 if d in all_diseases] or all_diseases

    selected_diseases = st.multiselect(
        "Select Diseases to Evaluate:",
        options=all_diseases,
        default=default_selection,
        key=A7_KEY + "disease_select",
        help="Select pathogens to evaluate real-time transmission velocity and sewershed viral concentrations."
    )

    if not selected_diseases:
        st.warning("Please select at least one disease.")
        return

    filtered_df = active_df[active_df["Disease Type"].isin(selected_diseases)].copy()
    profile_df = build_agent7_profile(
        filtered_df,
        jurisdiction=jurisdiction,
        lag_complete_weeks=1
    )

    st.subheader(
        "1) Outbreak Dynamics & Active Surveillance Matrix",
        help="Combines geographic spread containment classification with active CDC Center for Forecasting & Outbreak Analytics mathematical model outputs."
    )
    st.dataframe(profile_df, width="stretch")

    st.subheader(
        f"2) Real-Time Transmission & Wastewater Surveillance ({jur_label})",
        help="Surveillance feeds providing leading indicators of epidemic waves 1-2 weeks before hospital admission surges."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**Real-Time Transmission Velocity ($R_t$):**",
            help=(
                "Source: CDC CFA (Center for Forecasting & Analytics - Dataset 5dqz-y4ea). "
                "Daily Rt reproduction number. Rt > 1.0 indicates exponential transmission growth; "
                "Rt < 1.0 indicates epidemic wave contraction."
            )
        )
        try:
            rt_trends = get_rt_trends(jurisdiction=jurisdiction)
            st.line_chart(rt_trends)
        except Exception as e:
            st.caption(f"Rt chart loading: {e}")
            
    with c2:
        st.markdown(
            "**SARS-CoV-2 Wastewater Viral Load (copies/L):**",
            help=(
                "Source: CDC NWSS (National Wastewater Surveillance System - Dataset j9g8-acpt). "
                "7-day rolling median viral RNA gene copies per liter across municipal wastewater treatment facilities. "
                "Serves as an unbiased early indicator unaffected by healthcare-seeking behavior."
            )
        )
        try:
            ww_trends = get_wastewater_trends(jurisdiction=jurisdiction)
            st.line_chart(ww_trends)
        except Exception as e:
            st.caption(f"Wastewater chart loading: {e}")

    scored_cols = ["Geographic Spread Risk (Score)", "Effective Reproduction Number (Rt)", "Probability of Outbreak Growth"]
    tiers_df, dom_expl = mosdm_tiers(profile_df, criteria_cols=scored_cols)

    st.subheader(
        "3) MOSDM Prioritization (Pareto Dominance Tiers)",
        help="Identifies non-dominated pathogens exhibiting the fastest active epidemic expansion and widespread containment risk."
    )
    m1, m2 = st.columns([1, 1])
    with m1:
        st.markdown(
            "**Dominance Tiers (Tier 1 = Non-Dominated / Highest Urgent Outbreak Risk):**",
            help="Tier 1 indicates pathogens actively expanding in the community with high growth probabilities."
        )
        st.dataframe(tiers_df, width="stretch")
    with m2:
        st.markdown("**Dominance Explanations:**", help="Dominance logic across outbreak velocity dimensions.")
        if dom_expl:
            for e in dom_expl:
                st.markdown(f"- {e}")
        else:
            st.info("No strict dominance detected among selected diseases.")

    st.subheader(
        "4) SSDM Scenario Prioritization",
        help="Tie-breaking prioritizing rapid transmission velocity and community exponential growth."
    )
    scenario_names = [s["name"] for s in SCENARIOS_12]
    chosen_scenario = st.selectbox("Select Scenario:", scenario_names, index=2, key=A7_KEY + "scenario_select")
    
    scen_obj = next((s for s in SCENARIOS_12 if s["name"] == chosen_scenario), None)
    weights = scenario_weights_emphasis(scored_cols, scen_obj.get("emphasis", []))

    st.markdown("**Dynamic Criterion Weights Applied:**", help="Criterion weights applied to outbreak velocity metrics.")
    st.json({k: round(v, 4) for k, v in weights.items()})

    tier_blocks = []
    for t in sorted(tiers_df["Tier"].unique().tolist()):
        tier_diseases = tiers_df[tiers_df["Tier"] == t]["Disease Type"].tolist()
        block = profile_df[profile_df["Disease Type"].isin(tier_diseases)].copy()
        ranked_block = weighted_tiebreak(block, weights=weights, criteria_cols=scored_cols)
        ranked_block["Tier"] = t
        tier_blocks.append(ranked_block)

    final_ssdm_df = pd.concat(tier_blocks, ignore_index=True)
    final_ssdm_df["Overall Rank"] = range(1, len(final_ssdm_df) + 1)

    display_cols = ["Overall Rank", "Tier", "Disease Type", "weighted_score", "Geographic Spread Status", "Effective Reproduction Number (Rt)"]
    st.dataframe(final_ssdm_df[display_cols], width="stretch")

    st.divider()
    with st.expander("📚 Evidence Base & Research Citations", expanded=False):
        cits_df = load_citations(target_diseases=selected_diseases)
        if not cits_df.empty:
            st.dataframe(cits_df[["Target Disease", "Target Metric", "Source Organization", "Article/Report Title", "Publication/Update Date", "Original Live URL"]], width="stretch")