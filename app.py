import streamlit as st
import pandas as pd
from modules.status_banner import render_status_banner
from modules.agent_m_data import build_harmonized_long_table, get_active_decision_matrix
from modules.load_data import load_citations
from modules.live_connectors import (
    get_hospital_admission_trends,
    get_ed_visit_trends,
    get_meningitis_trends
)

st.set_page_config(
    page_title="SSDM Public Health Dashboard",
    page_icon="🩺",
    layout="wide"
)

st.title("SSDM Public Health Decision Support Dashboard")
st.caption("Stratified Multi-Criteria Prioritization across 8 Target Pathogens & 8 Analytical Domains")

# --- TOP INTERACTIVE DATA INGESTION STATUS BANNER ---
render_status_banner()

# --- SIDEBAR: JURISDICTION SELECTOR ---
st.sidebar.title("Surveillance Scope")
jurisdiction = st.sidebar.selectbox(
    "Active Jurisdiction:",
    options=["National", "MI"],
    format_func=lambda x: "US National" if x == "National" else "Michigan (MI)",
    key="global_jurisdiction",
    help=(
        "Selects the geographical boundary for surveillance feeds and baseline calculations. "
        "'US National' aggregates all 50 states and territories. 'Michigan (MI)' isolates "
        "state-level surveillance data."
    )
)
jur_label = "US National" if jurisdiction == "National" else "Michigan (MI)"
st.sidebar.caption(f"Active surveillance feeds set to **{jur_label}** (Lag: t-1 wk).")
st.sidebar.caption("Engine: updated MOSDM on every agent page; weights and "
                   "expert limits are elicited inside each agent (Section 3).")

st.sidebar.divider()
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page / Domain:",
    [
        "Overview & Master Dataset",
        "Agent 1: Epidemiological Burden",
        "Agent 2: Transmission & Susceptibility",
        "Agent 3: Healthcare System Impact",
        "Agent 4: Clinical Severity & Treatment",
        "Agent 5: Prevention & Control",
        "Agent 6: Vulnerable Populations",
        "Agent 7: Outbreak Dynamics & Risk",
        "Agent 8: Social & Economic Impact",
        "Scenarios & Probability",
        "SSDM Integration & Outputs",
        "About & Methodology"
    ],
    help="Navigate between the master surveillance overview, the 8 specialized domain agents, and technical methodology documentation."
)

if page == "Overview & Master Dataset":
    st.header(
        "Master Evidence Base & Surveillance Overview",
        help="Central repository integrating multi-year baseline epidemiology with automated weekly CDC surveillance feeds."
    )
    
    with st.expander("📖 How This Decision Support System Works (Plain-English Overview)", expanded=True):
        st.markdown("""
        This dashboard uses a **two-layer mathematical framework** to help public health officials prioritize resources without arbitrary score flattening:
        
        1. **Layer 1: Macro Strategic Baseline (365-Day Footprint)**
           * Measures the long-term, annual systemic burden across all 8 target pathogens (Incidence, Annual Hospitalizations, Case-Fatality Ratio, R0, and Direct Medical Costs).
        2. **Layer 2: Acute Real-Time Surveillance (Weekly Feeds)**
           * Tracks confirmed weekly hospital admissions, intensive care necessity, emergency room respiratory surges, and real-time viral transmission (Rt) for the active jurisdiction.
        3. **Multi-Objective Spatial Decision Making (MOSDM Tiers)**
           * Identifies **Pareto Dominance Tiers**. A pathogen is placed in **Tier 1** if no other disease completely overshadows it across all scored metrics, preserving multi-variable trade-offs.
        4. **Scenario-Based Spatial Decision Making (SSDM)**
           * Breaks ties within each dominance tier by applying policy weights tailored to 12 realistic public health emergencies (e.g., *Hospital Strain Winter*, *Vaccine Shortage*, *AMR Treatment Failure*).
        """)

    st.subheader(
        f"Active Surveillance Inflow ({jur_label})",
        help="Automated weekly time-series feeds pulled directly from CDC Socrata API endpoints. Data includes a standard 1-week reporting lag to guarantee completeness."
    )
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "**Weekly Respiratory Admissions (per 100k)**",
            help=(
                "Source: CDC NHSN (Dataset ua7e-t2fy). Measures weekly laboratory-confirmed new hospital admissions "
                "per 100,000 residents for COVID-19, Influenza, and RSV."
            )
        )
        try:
            hosp_trends = get_hospital_admission_trends(jurisdiction=jurisdiction)
            st.line_chart(hosp_trends)
        except Exception as e:
            st.caption(f"Hospital chart loading: {e}")
            
    with c2:
        st.markdown(
            "**Emergency Department Visit Share (%)**",
            help=(
                "Source: CDC NSSP (Dataset rdmq-nq56). Measures the percentage of emergency room visits "
                "diagnosed with COVID-19, Flu, or RSV."
            )
        )
        try:
            ed_trends = get_ed_visit_trends(jurisdiction=jurisdiction)
            st.line_chart(ed_trends)
        except Exception as e:
            st.caption(f"ED chart loading: {e}")
            
    with c3:
        st.markdown(
            "**Meningococcal Weekly Notifications (NNDSS)**",
            help=(
                "Source: CDC NNDSS (Dataset x9gk-5huc). Weekly confirmed/probable case counts for Meningococcal "
                "disease with a 4-week moving average."
            )
        )
        try:
            mening_trends = get_meningitis_trends(jurisdiction=jurisdiction)
            if not mening_trends.empty:
                st.line_chart(mening_trends)
            else:
                st.info("No NNDSS time-series records found.")
        except Exception as e:
            st.caption(f"Meningitis chart loading: {e}")

    st.subheader(
        f"Active Master Decision Matrix ({jur_label})",
        help="Harmonized multi-criteria dataset containing 365-day annual baselines merged with active weekly surveillance indicators."
    )
    df = get_active_decision_matrix(jurisdiction=jurisdiction)
    st.dataframe(df, width="stretch")

    st.divider()
    with st.expander("📚 Complete Evidence Base & Research Citations", expanded=False):
        cits_df = load_citations()
        st.dataframe(cits_df[["Target Disease", "Target Metric", "Source Organization", "Article/Report Title", "Publication/Update Date", "Original Live URL"]], width="stretch")

elif page == "Agent 1: Epidemiological Burden":
    from modules.agents import agent1
    agent1.render(jurisdiction=jurisdiction)

elif page == "Agent 2: Transmission & Susceptibility":
    from modules.agents import agent2
    agent2.render(jurisdiction=jurisdiction)

elif page == "Agent 3: Healthcare System Impact":
    from modules.agents import agent3
    agent3.render(jurisdiction=jurisdiction)

elif page == "Agent 4: Clinical Severity & Treatment":
    from modules.agents import agent4
    agent4.render(jurisdiction=jurisdiction)

elif page == "Agent 5: Prevention & Control":
    from modules.agents import agent5
    agent5.render(jurisdiction=jurisdiction)

elif page == "Agent 6: Vulnerable Populations":
    from modules.agents import agent6
    agent6.render(jurisdiction=jurisdiction)

elif page == "Agent 7: Outbreak Dynamics & Risk":
    from modules.agents import agent7
    agent7.render(jurisdiction=jurisdiction)

elif page == "Agent 8: Social & Economic Impact":
    from modules.agents import agent8
    agent8.render(jurisdiction=jurisdiction)

elif page == "Scenarios & Probability":
    from modules import scenario_view
    scenario_view.render(jurisdiction=jurisdiction)

elif page == "SSDM Integration & Outputs":
    from modules import ssdm_view
    ssdm_view.render(jurisdiction=jurisdiction)

elif page == "About & Methodology":
    from modules import about
    about.render()