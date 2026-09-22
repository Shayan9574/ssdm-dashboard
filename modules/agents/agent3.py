import streamlit as st
import pandas as pd
from modules.agent_m_data import get_active_decision_matrix
from modules.load_data import load_citations
from modules.live_connectors import get_ed_visit_trends, get_hospital_occupancy_trends
from modules.scenarios import SCENARIOS_12
from modules.agents.agent3_healthcare_impact import (
    AGENT_TITLE,
    SUBCRITERIA,
    build_agent3_profile,
    mosdm_tiers,
    weighted_tiebreak,
    scenario_weights_emphasis,
)

A3_KEY = "A3_"

def render(jurisdiction: str = "National"):
    jur_label = "US National" if jurisdiction == "National" else "Michigan (MI)"
    st.header(
        AGENT_TITLE,
        help="Tracks hospital bed occupancy, average length of stay (ALOS), and emergency department volume to assess acute care surge capacity."
    )
    st.caption(f"Inpatient bed duration, ICU surge pressure, and emergency department utilization for **{jur_label}**.")

    with st.expander("💡 What You Are Looking At (Plain-English Guide)", expanded=False):
        st.markdown("""
        * **Average Length of Stay (Days):** How many consecutive days a patient occupies an inpatient hospital bed. Longer stays directly tie up facility capacity.
        * **Emergency Department (ED) Visit Rate (%):** What percentage of emergency room visits are caused by acute respiratory symptoms. Acts as an early warning for hospital admission surges.
        * **Inpatient & ICU Bed Occupancy (%):** The overall percentage of licensed hospital and intensive care beds currently in use across the jurisdiction.
        """)

    active_df = get_active_decision_matrix(jurisdiction=jurisdiction)
    
    all_diseases = active_df["Disease Type"].tolist()
    core_4 = ["Influenza", "COVID-19", "Respiratory Syncytial Virus (RSV)", "Meningococcal Disease (Meningitis)"]
    default_selection = [d for d in core_4 if d in all_diseases] or all_diseases

    selected_diseases = st.multiselect(
        "Select Diseases to Evaluate:",
        options=all_diseases,
        default=default_selection,
        key=A3_KEY + "disease_select",
        help="Select pathogens to evaluate hospital bed bottlenecks and emergency department surge strain."
    )

    if not selected_diseases:
        st.warning("Please select at least one disease.")
        return

    filtered_df = active_df[active_df["Disease Type"].isin(selected_diseases)].copy()
    profile_df = build_agent3_profile(filtered_df)

    st.subheader(
        "1) Healthcare System Baseline Evidence Matrix",
        help="Strategic baseline values measuring average inpatient duration of stay and annual emergency department utilization."
    )
    st.dataframe(profile_df, width="stretch")

    st.subheader(
        f"2) Real-Time Emergency Department & Bed Strain ({jur_label})",
        help="Weekly live monitoring of syndromic emergency department encounters and jurisdictional hospital capacity occupancy (CDC NSSP & NHSN)."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**Weekly ED Syndromic Visit Share (%):**",
            help="Weekly percentage of emergency department visits attributed to acute viral infections (CDC NSSP)."
        )
        try:
            ed_trends = get_ed_visit_trends(jurisdiction=jurisdiction)
            st.line_chart(ed_trends)
        except Exception as e:
            st.caption(f"ED chart loading: {e}")
            
    with c2:
        st.markdown(
            "**Hospital Capacity Occupancy Trends (%):**",
            help="Weekly total licensed inpatient and adult ICU bed occupancy percentage across the jurisdiction (CDC NHSN)."
        )
        try:
            occ_trends = get_hospital_occupancy_trends(jurisdiction=jurisdiction)
            st.line_chart(occ_trends)
        except Exception as e:
            st.caption(f"Occupancy chart loading: {e}")

    st.subheader(
        "3) MOSDM Prioritization (Pareto Dominance Tiers)",
        help="Identifies non-dominated pathogens causing the greatest facility bottlenecks and emergency department volume."
    )
    numeric_cost_cols = [s["name"] for s in SUBCRITERIA if s["kind"] == "cost"]
    tiers_df, dom_expl = mosdm_tiers(profile_df, criteria_cols=numeric_cost_cols)

    m1, m2 = st.columns([1, 1])
    with m1:
        st.markdown(
            "**Dominance Tiers (Tier 1 = Non-Dominated / Highest Healthcare System Strain):**",
            help="Pathogens creating unmitigated pressure on inpatient beds and emergency departments."
        )
        st.dataframe(tiers_df, width="stretch")
    with m2:
        st.markdown("**Dominance Explanations:**", help="Dominance logic across healthcare system metrics.")
        if dom_expl:
            for e in dom_expl:
                st.markdown(f"- {e}")
        else:
            st.info("No strict dominance detected among selected diseases.")

    st.subheader(
        "4) SSDM Scenario-Weighted Tie-Breaking",
        help="Tie-breaking prioritizing inpatient stay duration, intensive care bed utilization, and emergency room overcrowding."
    )
    scenario_names = [s["name"] for s in SCENARIOS_12]
    chosen_scenario = st.selectbox("Select Scenario:", scenario_names, index=1, key=A3_KEY + "scenario_select")
    
    scen_obj = next((s for s in SCENARIOS_12 if s["name"] == chosen_scenario), None)
    weights = scenario_weights_emphasis(numeric_cost_cols, scen_obj.get("emphasis", []))

    st.markdown("**Dynamic Criterion Weights Applied:**", help="Scenario weights applied to healthcare strain criteria.")
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