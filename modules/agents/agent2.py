import streamlit as st
import pandas as pd
from modules.agent_m_data import get_active_decision_matrix
from modules.load_data import load_citations
from modules.live_connectors import get_test_positivity_trends
from modules.scenarios import SCENARIOS_12
from modules.agents.agent2_transmission import (
    AGENT_TITLE,
    SUBCRITERIA,
    build_agent2_profile,
    mosdm_tiers,
    weighted_tiebreak,
    scenario_weights_emphasis,
)

A2_KEY = "A2_"

def render(jurisdiction: str = "National"):
    jur_label = "US National" if jurisdiction == "National" else "Michigan (MI)"
    st.header(
        AGENT_TITLE,
        help="Evaluates inherent biological contagiousness (R0), natural antibody protection, asymptomatic spread rates, and household secondary attack rates."
    )
    st.caption(f"Contagion speed, antibody protection, and testing positivity for **{jur_label}**.")

    with st.expander("💡 What You Are Looking At (Plain-English Guide)", expanded=False):
        st.markdown("""
        * **Basic Reproduction Number (R0):** On average, how many people one sick person infects in an entirely susceptible population. Measures pure biological contagiousness.
        * **Natural Infection Seroprevalence (%):** What percentage of the population already has antibodies from past infection. Higher seroprevalence acts as a protective shield.
        * **Test Positivity Rate (%):** Of all laboratory PCR tests performed, what percentage return positive. Higher positivity indicates high community transmission and potential under-testing.
        * **Asymptomatic Transmission Rate (%):** What percentage of spread occurs from individuals who feel completely healthy and show no symptoms.
        * **Household Secondary Attack Rate (%):** How likely the virus is to spread from one family member to another inside the home.
        """)

    active_df = get_active_decision_matrix(jurisdiction=jurisdiction)
    
    all_diseases = active_df["Disease Type"].tolist()
    core_4 = ["Influenza", "COVID-19", "Respiratory Syncytial Virus (RSV)", "Meningococcal Disease (Meningitis)"]
    default_selection = [d for d in core_4 if d in all_diseases] or all_diseases

    selected_diseases = st.multiselect(
        "Select Diseases to Evaluate:",
        options=all_diseases,
        default=default_selection,
        key=A2_KEY + "disease_select",
        help="Select pathogens to evaluate community transmission dynamics and population immunity."
    )

    if not selected_diseases:
        st.warning("Please select at least one disease.")
        return

    filtered_df = active_df[active_df["Disease Type"].isin(selected_diseases)].copy()
    profile_df = build_agent2_profile(filtered_df)

    st.subheader(
        "1) Baseline Transmission Matrix",
        help="Strategic baseline values measuring reproductive speed, seroprevalence, asymptomatic proportion, and household secondary transmission."
    )
    st.dataframe(profile_df, width="stretch")

    st.subheader(
        "2) Real-Time Laboratory PCR Test Positivity Trends (%)",
        help="Source: CDC NREVSS (Dataset seuz-s2cv). Weekly percentage of laboratory nasal swabs testing positive across reporting clinical laboratories."
    )
    try:
        pos_trends = get_test_positivity_trends()
        if not pos_trends.empty:
            st.line_chart(pos_trends)
        else:
            st.info("No positivity time-series data available.")
    except Exception as e:
        st.caption(f"Positivity chart loading: {e}")

    scoring_df = profile_df.copy()
    cost_criteria_cols = []
    
    for sc in SUBCRITERIA:
        name = sc["name"]
        if sc["kind"] == "benefit":
            scoring_df[name + " (Cost Space)"] = -scoring_df[name]
            cost_criteria_cols.append(name + " (Cost Space)")
        else:
            cost_criteria_cols.append(name)

    st.subheader(
        "3) MOSDM Prioritization (Pareto Dominance Tiers)",
        help="Evaluates transmission risk in Cost Space. Natural seroprevalence is inverted so higher numbers uniformly represent greater unmitigated spread risk."
    )
    tiers_df, dom_expl = mosdm_tiers(scoring_df, criteria_cols=cost_criteria_cols)
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            "**Dominance Tiers (Tier 1 = Non-Dominated / Highest Transmission Risk):**",
            help="Tier 1 pathogens represent the highest combined transmission velocity and immune-escape potential."
        )
        st.dataframe(tiers_df, width="stretch")
    with c2:
        st.markdown(
            "**Dominance Explanations:**",
            help="Explains which pathogen strictly dominates another across all transmission criteria."
        )
        if dom_expl:
            for e in dom_expl:
                st.markdown(f"- {e}")
        else:
            st.info("No strict dominance detected among selected diseases.")

    st.subheader(
        "4) SSDM Scenario-Weighted Tie-Breaking",
        help="Ranks diseases within each tier based on scenario weights emphasizing contagion speed, household spread, and asymptomatic shedding."
    )
    scenario_names = [s["name"] for s in SCENARIOS_12]
    chosen_scenario = st.selectbox("Select Scenario:", scenario_names, index=3, key=A2_KEY + "scenario_select")
    
    scen_obj = next((s for s in SCENARIOS_12 if s["name"] == chosen_scenario), None)
    weights = scenario_weights_emphasis(cost_criteria_cols, scen_obj.get("emphasis", []))

    st.markdown("**Dynamic Criterion Weights Applied:**", help="Criterion weights tailored to transmission scenarios.")
    st.json({k: round(v, 4) for k, v in weights.items()})

    tier_blocks = []
    for t in sorted(tiers_df["Tier"].unique().tolist()):
        tier_diseases = tiers_df[tiers_df["Tier"] == t]["Disease Type"].tolist()
        block = scoring_df[scoring_df["Disease Type"].isin(tier_diseases)].copy()
        ranked_block = weighted_tiebreak(block, weights=weights, criteria_cols=cost_criteria_cols)
        ranked_block["Tier"] = t
        tier_blocks.append(ranked_block)

    final_ssdm_df = pd.concat(tier_blocks, ignore_index=True)
    final_ssdm_df["Overall Rank"] = range(1, len(final_ssdm_df) + 1)

    display_cols = ["Overall Rank", "Tier", "Disease Type", "weighted_score"] + [s["name"] for s in SUBCRITERIA]
    st.dataframe(final_ssdm_df[display_cols], width="stretch")

    st.divider()
    with st.expander("📚 Evidence Base & Research Citations", expanded=False):
        cits_df = load_citations(target_diseases=selected_diseases)
        if not cits_df.empty:
            st.dataframe(cits_df[["Target Disease", "Target Metric", "Source Organization", "Article/Report Title", "Publication/Update Date", "Original Live URL"]], width="stretch")