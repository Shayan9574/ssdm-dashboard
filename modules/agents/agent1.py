import streamlit as st
import pandas as pd
from modules.agent_m_data import get_active_decision_matrix
from modules.load_data import load_citations
from modules.live_connectors import get_hospital_admission_trends, get_meningitis_trends
from modules.scenarios import SCENARIOS_12
from modules.agents.agent1_epidemiology import (
    AGENT_TITLE,
    SUBCRITERIA,
    build_agent1_profile,
    mosdm_tiers,
    weighted_tiebreak,
    scenario_weights_emphasis,
)

A1_KEY = "A1_"

def render(jurisdiction: str = "National"):
    jur_label = "US National" if jurisdiction == "National" else "Michigan (MI)"
    st.header(
        AGENT_TITLE,
        help="Evaluates total annual population morbidity, hospital admission density, intensive care utilization, and clinical lethality."
    )
    st.caption(f"Stratified analysis of annual disease volume, hospitalizations, intensive care necessity, and lethality for **{jur_label}**.")

    with st.expander("💡 What You Are Looking At (Plain-English Guide)", expanded=False):
        st.markdown("""
        * **Incidence Rate:** How many people catch the disease per 100,000 population each year. Measures overall community disease volume.
        * **Hospitalization Rate:** How many people become sick enough to be admitted to a hospital per 100,000 population.
        * **ICU Rate (% of Hospitalized):** Of those admitted to the hospital, what percentage need intensive care. Measures disease severity and acute care pressure.
        * **Case-Fatality Ratio (CFR %):** What percentage of diagnosed patients die from the infection. Measures clinical lethality (distinct from population mortality rate).
        * **Pareto Dominance (MOSDM Tiers):** Ranks diseases without flattening metrics into an arbitrary single score. **Tier 1** represents diseases that have higher burden across multiple dimensions and cannot be ruled out by any single lesser disease.
        """)

    active_df = get_active_decision_matrix(jurisdiction=jurisdiction)
    
    all_diseases = active_df["Disease Type"].tolist()
    core_4 = ["Influenza", "COVID-19", "Respiratory Syncytial Virus (RSV)", "Meningococcal Disease (Meningitis)"]
    default_selection = [d for d in core_4 if d in all_diseases] or all_diseases

    selected_diseases = st.multiselect(
        "Select Diseases to Evaluate:",
        options=all_diseases,
        default=default_selection,
        key=A1_KEY + "disease_select",
        help="Choose any combination of the 8 tracked pathogens to compute localized Pareto dominance tiers and weighted tie-breakers."
    )

    if not selected_diseases:
        st.warning("Please select at least one disease.")
        return

    filtered_df = active_df[active_df["Disease Type"].isin(selected_diseases)].copy()
    profile_df = build_agent1_profile(filtered_df)

    # 1. Macro Baseline Matrix
    st.subheader(
        "1) Macro Annual Baseline Evidence Matrix",
        help="Standardized 365-day strategic baseline figures derived from multi-year CDC MMWR annual summaries and national registry audits."
    )
    st.dataframe(profile_df, width="stretch")

    # 2. Acute Surveillance Feed Overlay & Trend Charts
    st.subheader(
        f"2) Real-Time Weekly Surveillance Inflow ({jur_label})",
        help="Dynamic indicators reflecting active strain on healthcare facilities over the most recent reporting week (t-1 lag)."
    )
    surv_cols = ["Disease Type", "live_weekly_hosp_adm_per_100k", "live_icu_share_pct"]
    surv_df = filtered_df[surv_cols].rename(columns={
        "live_weekly_hosp_adm_per_100k": "Weekly New Admissions (per 100k)",
        "live_icu_share_pct": "Current ICU Share (% of Hosp)"
    })
    st.dataframe(surv_df, width="stretch")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown(
            "**Weekly Respiratory Admissions (per 100k):**",
            help="Weekly new admissions per 100k residents for viral respiratory pathogens (CDC NHSN)."
        )
        try:
            hosp_trends = get_hospital_admission_trends(jurisdiction=jurisdiction)
            if not hosp_trends.empty:
                st.line_chart(hosp_trends)
            else:
                st.info("No time-series admission data available.")
        except Exception as e:
            st.caption(f"Hospital chart loading: {e}")
            
    with g2:
        st.markdown(
            "**Meningococcal Disease Weekly Notifications (NNDSS):**",
            help="Weekly confirmed/probable case reports for Meningococcal disease alongside a 4-week smoothing average (CDC NNDSS)."
        )
        try:
            mening_trends = get_meningitis_trends(jurisdiction=jurisdiction)
            if not mening_trends.empty:
                st.line_chart(mening_trends)
            else:
                st.info("No NNDSS time-series records available.")
        except Exception as e:
            st.caption(f"Meningitis chart loading: {e}")

    # 3. MOSDM Prioritization
    st.subheader(
        "3) MOSDM Prioritization (Pareto Dominance Tiers)",
        help=(
            "Multi-Objective Spatial Decision Making (MOSDM) identifies non-dominated alternatives in Cost Space. "
            "A disease is placed in Tier 1 if no other disease exhibits higher burden across all evaluated dimensions."
        )
    )
    numeric_cost_cols = [s["name"] for s in SUBCRITERIA if s["kind"] == "cost"]
    tiers_df, dom_expl = mosdm_tiers(profile_df, criteria_cols=numeric_cost_cols)
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            "**Dominance Tiers (Tier 1 = Non-Dominated / Highest Strategic Priority):**",
            help="Tier 1 diseases represent non-dominated threats requiring immediate macro resource prioritization."
        )
        st.dataframe(tiers_df, width="stretch")
    with c2:
        st.markdown(
            "**Dominance Explanations:**",
            help="Mathematical rationale identifying exactly which pathogen overshadows another across all scored metrics."
        )
        if dom_expl:
            for e in dom_expl:
                st.markdown(f"- {e}")
        else:
            st.info("No strict dominance detected among selected diseases.")

    # 4. SSDM Scenario Analysis
    st.subheader(
        "4) SSDM Scenario-Weighted Tie-Breaking",
        help="Breaks ranking ties within each Pareto tier using multi-criteria weighted sums calibrated to 12 operational public health emergency scenarios."
    )
    scenario_names = [s["name"] for s in SCENARIOS_12]
    chosen_scenario = st.selectbox(
        "Select Scenario:",
        scenario_names,
        index=0,
        key=A1_KEY + "scenario_select",
        help="Select an operational context (e.g., Hospital Surge, Vaccine Shortage) to dynamically re-weight evaluation criteria."
    )
    
    scen_obj = next((s for s in SCENARIOS_12 if s["name"] == chosen_scenario), None)
    weights = scenario_weights_emphasis(numeric_cost_cols, scen_obj.get("emphasis", []))

    st.markdown(
        "**Dynamic Criterion Weights Applied:**",
        help="Linear weights assigned to each metric based on the selected operational scenario."
    )
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

    # 5. Evidence Base & Citations
    st.divider()
    with st.expander("📚 Evidence Base & Research Citations", expanded=False):
        cits_df = load_citations(target_diseases=selected_diseases)
        if not cits_df.empty:
            st.dataframe(cits_df[["Target Disease", "Target Metric", "Source Organization", "Article/Report Title", "Publication/Update Date", "Original Live URL"]], width="stretch")
        else:
            st.info("No specific citation records found for selected diseases.")