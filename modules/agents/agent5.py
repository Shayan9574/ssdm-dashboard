import streamlit as st
import pandas as pd
from modules.agent_m_data import get_active_decision_matrix
from modules.load_data import load_citations
from modules.scenarios import SCENARIOS_12
from modules.agents.agent5_prevention import (
    AGENT_TITLE,
    SUBCRITERIA,
    build_agent5_profile,
    mosdm_tiers_benefit,
    weighted_tiebreak_benefit,
    scenario_weights_emphasis,
)

A5_KEY = "A5_"

def render(jurisdiction: str = "National"):
    jur_label = "US National" if jurisdiction == "National" else "Michigan (MI)"
    st.header(
        AGENT_TITLE,
        help="Evaluates vaccine effectiveness, population coverage, non-pharmaceutical intervention (NPI) efficacy, and high-risk group uptake."
    )
    st.caption(f"Vaccine availability, immunization coverage, and non-pharmaceutical intervention effectiveness ({jur_label}).")

    with st.expander("💡 What You Are Looking At (Plain-English Guide)", expanded=False):
        st.markdown("""
        * **Vaccine Available (Yes/No):** Whether a safe, FDA-authorized immunization is commercially accessible.
        * **Vaccine Effectiveness (%):** How effectively the vaccine prevents severe hospitalization and death.
        * **National Vaccine Coverage (%):** What percentage of the eligible population has received the vaccine.
        * **NPI Transmission Reduction (%):** How effectively Non-Pharmaceutical Interventions reduce spread.
        * **Target Demographic Vaccine Uptake (%):** Immunization rates among the most vulnerable high-risk groups.
        * **Benefit Space:** In this agent, **higher numbers are better**. Tier 1 represents diseases with the strongest prevention tools.
        """)

    active_df = get_active_decision_matrix(jurisdiction=jurisdiction)
    
    all_diseases = active_df["Disease Type"].tolist()
    core_4 = ["Influenza", "COVID-19", "Respiratory Syncytial Virus (RSV)", "Meningococcal Disease (Meningitis)"]
    default_selection = [d for d in core_4 if d in all_diseases] or all_diseases

    selected_diseases = st.multiselect(
        "Select Diseases to Evaluate:",
        options=all_diseases,
        default=default_selection,
        key=A5_KEY + "disease_select",
        help="Select pathogens to evaluate vaccine protection capacity and community intervention coverage."
    )

    if not selected_diseases:
        st.warning("Please select at least one disease.")
        return

    filtered_df = active_df[active_df["Disease Type"].isin(selected_diseases)].copy()
    profile_df = build_agent5_profile(filtered_df)

    st.subheader(
        "1) Baseline Prevention & Control Evidence Matrix",
        help="Baseline metrics for vaccine efficacy, population coverage, and non-pharmaceutical intervention impact."
    )
    st.dataframe(profile_df, width="stretch")

    numeric_benefit_cols = [s["name"] for s in SUBCRITERIA]
    tiers_df, dom_expl = mosdm_tiers_benefit(profile_df, criteria_cols=numeric_benefit_cols)

    st.subheader(
        "2) MOSDM Prioritization (Pareto Dominance Tiers in Benefit Space)",
        help="Evaluated in Benefit Space where higher values indicate superior public health protection and control readiness. Tier 1 = highest prevention capacity."
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            "**Dominance Tiers (Tier 1 = Highest Prevention Capacity):**",
            help="Tier 1 pathogens have robust vaccines and established community uptake."
        )
        st.dataframe(tiers_df, width="stretch")
    with c2:
        st.markdown("**Dominance Explanations:**", help="Dominance logic across prevention metrics.")
        if dom_expl:
            for e in dom_expl:
                st.markdown(f"- {e}")
        else:
            st.info("No strict dominance detected among selected diseases.")

    st.subheader(
        "3) SSDM Scenario-Weighted Tie-Breaking",
        help="Tie-breaking prioritizing vaccine coverage, effectiveness, and non-pharmaceutical interventions."
    )
    scenario_names = [s["name"] for s in SCENARIOS_12]
    chosen_scenario = st.selectbox("Select Scenario:", scenario_names, index=1, key=A5_KEY + "scenario_select")
    
    scen_obj = next((s for s in SCENARIOS_12 if s["name"] == chosen_scenario), None)
    weights = scenario_weights_emphasis(numeric_benefit_cols, scen_obj.get("emphasis", []))

    st.markdown("**Dynamic Criterion Weights Applied:**", help="Criterion weights applied to prevention capacity.")
    st.json({k: round(v, 4) for k, v in weights.items()})

    tier_blocks = []
    for t in sorted(tiers_df["Tier"].unique().tolist()):
        tier_diseases = tiers_df[tiers_df["Tier"] == t]["Disease Type"].tolist()
        block = profile_df[profile_df["Disease Type"].isin(tier_diseases)].copy()
        ranked_block = weighted_tiebreak_benefit(block, weights=weights, criteria_cols=numeric_benefit_cols)
        ranked_block["Tier"] = t
        tier_blocks.append(ranked_block)

    final_ssdm_df = pd.concat(tier_blocks, ignore_index=True)
    final_ssdm_df["Overall Rank"] = range(1, len(final_ssdm_df) + 1)

    display_cols = ["Overall Rank", "Tier", "Disease Type", "weighted_score"] + numeric_benefit_cols
    st.dataframe(final_ssdm_df[display_cols], width="stretch")

    st.divider()
    with st.expander("📚 Evidence Base & Research Citations", expanded=False):
        cits_df = load_citations(target_diseases=selected_diseases)
        if not cits_df.empty:
            st.dataframe(cits_df[["Target Disease", "Target Metric", "Source Organization", "Article/Report Title", "Publication/Update Date", "Original Live URL"]], width="stretch")