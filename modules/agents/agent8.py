import streamlit as st
import pandas as pd
from modules.agent_m_data import get_active_decision_matrix
from modules.load_data import load_citations
from modules.scenarios import SCENARIOS_12
from modules.agents.agent8_social import (
    AGENT_TITLE,
    build_agent8_profile,
    mosdm_tiers,
    weighted_tiebreak,
)

A8_KEY = "A8_"

def render(jurisdiction: str = "National"):
    jur_label = "US National" if jurisdiction == "National" else "Michigan (MI)"
    st.header(
        AGENT_TITLE,
        help="Evaluates structural social vulnerability drivers (CDC SVI themes) and total annual direct medical expenditures (in billions USD)."
    )
    st.caption(f"Social vulnerability drivers, structural inequity multipliers, and annual direct medical costs ({jur_label}).")

    with st.expander("💡 What You Are Looking At (Plain-English Guide)", expanded=False):
        st.markdown("""
        * **Primary SVI Vulnerability Driver:** The dominant Social Vulnerability Index factor amplifying disease transmission.
        * **Estimated Annual Direct Medical Cost ($B):** Total financial expenditure spent each year on direct medical care (in billions USD).
        """)

    active_df = get_active_decision_matrix(jurisdiction=jurisdiction)
    
    all_diseases = active_df["Disease Type"].tolist()
    core_4 = ["Influenza", "COVID-19", "Respiratory Syncytial Virus (RSV)", "Meningococcal Disease (Meningitis)"]
    default_selection = [d for d in core_4 if d in all_diseases] or all_diseases

    selected_diseases = st.multiselect(
        "Select Diseases to Evaluate:",
        options=all_diseases,
        default=default_selection,
        key=A8_KEY + "disease_select",
        help="Select pathogens to evaluate economic healthcare expenditures and social vulnerability factors."
    )

    if not selected_diseases:
        st.warning("Please select at least one disease.")
        return

    filtered_df = active_df[active_df["Disease Type"].isin(selected_diseases)].copy()
    profile_df = build_agent8_profile(filtered_df)

    st.subheader(
        "1) Baseline Social & Economic Burden Matrix",
        help="Identifies primary Social Vulnerability Index (SVI) drivers and annual direct medical costs in billions USD."
    )
    display_profile = profile_df[["Disease Type", "Primary SVI Vulnerability Driver", "Estimated Annual Direct Medical Cost ($B)"]]
    st.dataframe(display_profile, width="stretch")

    scored_cols = ["Estimated Annual Direct Medical Cost ($B)"]
    tiers_df, dom_expl = mosdm_tiers(profile_df, criteria_cols=scored_cols)

    st.subheader(
        "2) MOSDM Prioritization (Pareto Dominance Tiers)",
        help="Ranks pathogens in Cost Space by total annual direct medical burden."
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            "**Dominance Tiers (Tier 1 = Highest Annual Medical Cost):**",
            help="Tier 1 identifies pathogens generating the largest direct economic healthcare costs."
        )
        st.dataframe(tiers_df, width="stretch")
    with c2:
        st.markdown("**Dominance Explanations:**", help="Dominance logic across direct economic costs.")
        if dom_expl:
            for e in dom_expl:
                st.markdown(f"- {e}")
        else:
            st.info("No strict dominance detected among selected diseases.")

    st.subheader(
        "3) SSDM Scenario Prioritization",
        help="Applies economic and healthcare cost reduction weighting."
    )
    scenario_names = [s["name"] for s in SCENARIOS_12]
    chosen_scenario = st.selectbox("Select Scenario:", scenario_names, index=5, key=A8_KEY + "scenario_select")
    
    weights = {c: 1.0 / len(scored_cols) for c in scored_cols}

    tier_blocks = []
    for t in sorted(tiers_df["Tier"].unique().tolist()):
        tier_diseases = tiers_df[tiers_df["Tier"] == t]["Disease Type"].tolist()
        block = profile_df[profile_df["Disease Type"].isin(tier_diseases)].copy()
        ranked_block = weighted_tiebreak(block, weights=weights, criteria_cols=scored_cols)
        ranked_block["Tier"] = t
        tier_blocks.append(ranked_block)

    final_ssdm_df = pd.concat(tier_blocks, ignore_index=True)
    final_ssdm_df["Overall Rank"] = range(1, len(final_ssdm_df) + 1)

    display_cols = ["Overall Rank", "Tier", "Disease Type", "Estimated Annual Direct Medical Cost ($B)", "Primary SVI Vulnerability Driver"]
    st.dataframe(final_ssdm_df[display_cols], width="stretch")

    st.divider()
    with st.expander("📚 Evidence Base & Research Citations", expanded=False):
        cits_df = load_citations(target_diseases=selected_diseases)
        if not cits_df.empty:
            st.dataframe(cits_df[["Target Disease", "Target Metric", "Source Organization", "Article/Report Title", "Publication/Update Date", "Original Live URL"]], width="stretch")