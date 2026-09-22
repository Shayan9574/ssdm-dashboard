import streamlit as st
import pandas as pd
from modules.agent_m_data import get_active_decision_matrix
from modules.load_data import load_citations
from modules.scenarios import SCENARIOS_12
from modules.agents.agent6_equity import (
    AGENT_TITLE,
    SUBCRITERIA,
    build_agent6_profile,
    mosdm_tiers,
    weighted_tiebreak,
    scenario_weights_emphasis,
)

A6_KEY = "A6_"

def render(jurisdiction: str = "National"):
    jur_label = "US National" if jurisdiction == "National" else "Michigan (MI)"
    st.header(
        AGENT_TITLE,
        help="Evaluates disproportionate hospitalization rates in pediatric (<5) and geriatric (65+) populations, along with racial and ethnic health disparity ratios."
    )
    st.caption(f"Pediatric vulnerability, geriatric hospitalization burden, and racial/ethnic disparity rate ratios ({jur_label}).")

    with st.expander("💡 What You Are Looking At (Plain-English Guide)", expanded=False):
        st.markdown("""
        * **Pediatric Hospitalization Rate (<5):** Hospital admissions per 100,000 children under age 5.
        * **Geriatric Hospitalization Rate (65+):** Hospital admissions per 100,000 adults aged 65 and older.
        * **Highest Racial Disparity Rate Ratio:** The burden ratio comparing the most impacted racial group against the least impacted group.
        """)

    active_df = get_active_decision_matrix(jurisdiction=jurisdiction)
    
    all_diseases = active_df["Disease Type"].tolist()
    core_4 = ["Influenza", "COVID-19", "Respiratory Syncytial Virus (RSV)", "Meningococcal Disease (Meningitis)"]
    default_selection = [d for d in core_4 if d in all_diseases] or all_diseases

    selected_diseases = st.multiselect(
        "Select Diseases to Evaluate:",
        options=all_diseases,
        default=default_selection,
        key=A6_KEY + "disease_select",
        help="Select pathogens to evaluate age-stratified morbidity and systemic health inequities."
    )

    if not selected_diseases:
        st.warning("Please select at least one disease.")
        return

    filtered_df = active_df[active_df["Disease Type"].isin(selected_diseases)].copy()
    profile_df = build_agent6_profile(filtered_df)

    st.subheader(
        "1) Baseline Vulnerable Population Matrix",
        help="Baseline metrics tracking pediatric admissions (<5), geriatric admissions (65+), and racial disparity rate ratios."
    )
    st.dataframe(profile_df, width="stretch")

    numeric_cost_cols = [s["name"] for s in SUBCRITERIA]
    tiers_df, dom_expl = mosdm_tiers(profile_df, criteria_cols=numeric_cost_cols)

    st.subheader(
        "2) MOSDM Prioritization (Pareto Dominance Tiers)",
        help="Identifies non-dominated pathogens causing the highest disproportionate harm to high-risk demographic groups."
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            "**Dominance Tiers (Tier 1 = Non-Dominated / Highest Inequity & Burden):**",
            help="Tier 1 diseases generate the highest combined burden across children, seniors, and historically marginalized groups."
        )
        st.dataframe(tiers_df, width="stretch")
    with c2:
        st.markdown("**Dominance Explanations:**", help="Dominance logic across demographic vulnerability dimensions.")
        if dom_expl:
            for e in dom_expl:
                st.markdown(f"- {e}")
        else:
            st.info("No strict dominance detected among selected diseases.")

    st.subheader(
        "3) SSDM Scenario-Weighted Tie-Breaking",
        help="Tie-breaking prioritizing pediatric critical care surges, geriatric lethality, and health disparity reduction."
    )
    scenario_names = [s["name"] for s in SCENARIOS_12]
    chosen_scenario = st.selectbox("Select Scenario:", scenario_names, index=4, key=A6_KEY + "scenario_select")
    
    scen_obj = next((s for s in SCENARIOS_12 if s["name"] == chosen_scenario), None)
    weights = scenario_weights_emphasis(numeric_cost_cols, scen_obj.get("emphasis", []))

    st.markdown("**Dynamic Criterion Weights Applied:**", help="Criterion weights applied to equity and vulnerability metrics.")
    st.json({k: round(v, 4) for k, v in weights.items()})

    tier_blocks = []
    for t in sorted(tiers_df["Tier"].unique().tolist()):
        tier_diseases = tiers_df[tiers_df["Tier"] == t]["Disease Type"].tolist()
        block = profile_df[profile_df["Disease Type"].isin(tier_diseases)].copy()
        ranked_block = weighted_tiebreak(block, weights=weights, criteria_cols=numeric_cost_cols)
        ranked_block["Tier"] = t
        tier_blocks.append(ranked_block)

    final_ssdm_df = pd.concat(tier_blocks, ignore_index=True)
    final_ssdm_df["Overall Rank"] = range(1, len(final_ssdm_df) + 1)

    display_cols = ["Overall Rank", "Tier", "Disease Type", "weighted_score"] + numeric_cost_cols
    st.dataframe(final_ssdm_df[display_cols], width="stretch")

    st.divider()
    with st.expander("📚 Evidence Base & Research Citations", expanded=False):
        cits_df = load_citations(target_diseases=selected_diseases)
        if not cits_df.empty:
            st.dataframe(cits_df[["Target Disease", "Target Metric", "Source Organization", "Article/Report Title", "Publication/Update Date", "Original Live URL"]], width="stretch")