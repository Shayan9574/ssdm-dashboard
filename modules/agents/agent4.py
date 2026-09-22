import streamlit as st
import pandas as pd
from modules.agent_m_data import get_active_decision_matrix
from modules.load_data import load_citations
from modules.scenarios import SCENARIOS_12
from modules.agents.agent4_clinical import (
    AGENT_TITLE,
    SUBCRITERIA,
    build_agent4_profile,
    build_cost_space_matrix,
    mosdm_tiers,
    weighted_tiebreak,
    scenario_weights_emphasis,
)

A4_KEY = "A4_"

def render(jurisdiction: str = "National"):
    jur_label = "US National" if jurisdiction == "National" else "Michigan (MI)"
    st.header(
        AGENT_TITLE,
        help="Evaluates clinical urgency: time to severe illness, therapeutic intervention windows, drug resistance prevalence, and long-term severe sequelae."
    )
    st.caption(f"Clinical deterioration timelines, therapeutic intervention windows, antimicrobial resistance, and severe complications ({jur_label}).")

    with st.expander("💡 What You Are Looking At (Plain-English Guide)", expanded=False):
        st.markdown("""
        * **Days from Onset to Severe Disease:** How quickly an infected patient declines into critical condition. Shorter times mean doctors have less time to intervene.
        * **Effective Treatment Window (Days):** The strict time limit from first symptoms during which antivirals or antibiotics must be administered.
        * **Standard Treatment Available:** Whether FDA-approved standard therapeutics exist (1 = Available, 0 = No standard protocol).
        * **Antimicrobial Resistance (AMR) Prevalence (%):** What percentage of circulating strains resist frontline treatments.
        * **Severe Complication Rate (%):** How often the infection leads to long-term organ damage or secondary post-acute syndromes.
        """)

    active_df = get_active_decision_matrix(jurisdiction=jurisdiction)
    
    all_diseases = active_df["Disease Type"].tolist()
    core_4 = ["Influenza", "COVID-19", "Respiratory Syncytial Virus (RSV)", "Meningococcal Disease (Meningitis)"]
    default_selection = [d for d in core_4 if d in all_diseases] or all_diseases

    selected_diseases = st.multiselect(
        "Select Diseases to Evaluate:",
        options=all_diseases,
        default=default_selection,
        key=A4_KEY + "disease_select",
        help="Select pathogens to evaluate clinical urgency, narrow treatment windows, and drug resistance risk."
    )

    if not selected_diseases:
        st.warning("Please select at least one disease.")
        return

    filtered_df = active_df[active_df["Disease Type"].isin(selected_diseases)].copy()
    profile_df = build_agent4_profile(filtered_df)

    st.subheader(
        "1) Baseline Clinical Evidence Matrix",
        help="Baseline metrics for clinical progression velocity, antiviral/antibiotic treatment windows, and complication frequencies."
    )
    st.dataframe(profile_df, width="stretch")

    cost_matrix_df, cost_criteria_cols = build_cost_space_matrix(profile_df)

    st.subheader(
        "2) MOSDM Prioritization (Pareto Dominance Tiers)",
        help="Evaluates clinical urgency in Inverted Cost Space. Narrower treatment windows and faster onset times are transformed into higher urgency cost scores."
    )
    tiers_df, dom_expl = mosdm_tiers(cost_matrix_df, criteria_cols=cost_criteria_cols)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            "**Dominance Tiers (Tier 1 = Non-Dominated / Highest Clinical Urgency):**",
            help="Tier 1 pathogens have rapid deterioration and narrow treatment windows (e.g., Meningococcal disease)."
        )
        st.dataframe(tiers_df, width="stretch")
    with c2:
        st.markdown("**Dominance Explanations:**", help="Dominance logic across clinical urgency dimensions.")
        if dom_expl:
            for e in dom_expl:
                st.markdown(f"- {e}")
        else:
            st.info("No strict dominance detected among selected diseases.")

    st.subheader(
        "3) SSDM Scenario-Weighted Tie-Breaking",
        help="Tie-breaking prioritizing rapid deterioration, antimicrobial resistance (AMR), and life-threatening complications."
    )
    scenario_names = [s["name"] for s in SCENARIOS_12]
    chosen_scenario = st.selectbox("Select Scenario:", scenario_names, index=9, key=A4_KEY + "scenario_select")
    
    scen_obj = next((s for s in SCENARIOS_12 if s["name"] == chosen_scenario), None)
    weights = scenario_weights_emphasis(cost_criteria_cols, scen_obj.get("emphasis", []))

    st.markdown("**Dynamic Criterion Weights Applied:**", help="Criterion weights applied to clinical severity metrics.")
    st.json({k: round(v, 4) for k, v in weights.items()})

    tier_blocks = []
    for t in sorted(tiers_df["Tier"].unique().tolist()):
        tier_diseases = tiers_df[tiers_df["Tier"] == t]["Disease Type"].tolist()
        block = cost_matrix_df[cost_matrix_df["Disease Type"].isin(tier_diseases)].copy()
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