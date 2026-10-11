import streamlit as st
import pandas as pd

def render():
    st.header(
        "About the SSDM Decision Support Framework",
        help="Technical documentation, mathematical formulation, surveillance ingestion pipelines, and analytical domain architecture."
    )
    st.caption("A multi-criteria mathematical prioritization system balancing long-term epidemiological baselines with real-time CDC surveillance streams.")

    # --- 1. EXECUTIVE SUMMARY & PURPOSE ---
    st.markdown("**1. System Overview & Core Purpose**")
    st.markdown(
        "Standard public health prioritization often collapses diverse epidemiological dimensions "
        "(high annual incidence versus low incidence, high lethality threats) into a single weighted average too early. "
        "The **SSDM Public Health Dashboard** keeps those dimensions apart: eight domain agents each stratify the "
        "pathogens with a **second generation MOSDM core** (graded attainment against criterion limits, tolerance "
        "tiers and a dominance guard), a **scenario layer (SSDM)** re-runs every agent under nine registered "
        "scenarios weighted by probability, and an integration layer combines the agents without ever letting the "
        "AI assistant alter a computed value."
    )

    # --- 2. TWO-LAYER ARCHITECTURE ---
    st.markdown("**2. Two-Layer Data Architecture**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "> **Layer 1: Macro Strategic Baseline (365-Day Footprint)**\n"
            "> * **Scope:** Evaluates long-term, structural burden across all 8 target pathogens.\n"
            "> * **Metrics:** Annual incidence, total hospitalizations, ICU admission rate, Case-Fatality Ratio (CFR), "
            "basic reproduction number ($R_0$), and direct medical costs ($B).\n"
            "> * **Provenance:** Extracted from multi-year CDC MMWR summaries, NHSN baselines, and peer-reviewed literature."
        )
    with c2:
        st.markdown(
            "> **Layer 2: Acute Real-Time Surveillance (Weekly Feeds)**\n"
            "> * **Scope:** Ingests weekly jurisdictional time-series to capture active outbreak surges.\n"
            "> * **Metrics:** Confirmed weekly hospital admissions per 100k, ED syndromic visit shares (%), "
            "laboratory PCR test positivity (%), wastewater viral concentrations, and daily transmission velocity ($R_t$).\n"
            "> * **Latency:** 1-week completed reporting lag ($t-1$) to eliminate provisional backfill bias."
        )

    # --- 3. MATHEMATICAL DECISION SCIENCE ---
    st.markdown("**3. Mathematical Formulations & Decision Science**")
    
    with st.expander("📐 MOSDM core: graded attainment and tolerance tiers", expanded=True):
        st.markdown(
            "Every criterion is oriented so that larger means more urgent (cost space; benefit criteria and the "
            "clinical time windows are inverted). Each value is converted to a **graded attainment** "
            "$\\tilde a_{ij} \\in [0,1]$ against the criterion limits, which come from expert limits when "
            "elicited, otherwise from historical surveillance quantiles, otherwise from the pool median. "
            "The selected default uses a fixed anchor $\\theta = 0.3$ (selected by a preregistered baseline study).\n\n"
            "$$K_i = \\sum_j w_j\\,\\tilde a_{ij}, \\qquad \\delta = \\min_{j:\\,w_j>0} w_j$$\n\n"
            "* **Tiering:** the pathogen with the largest weighted attainment $K_{\\max}$ enters Tier 1 together "
            "with every pathogen within the tolerance $\\delta$ of it; the tier is removed and the step repeats.\n"
            "* **Dominance guard:** a pathogen that is dominated on all shared observed criteria cannot be placed "
            "above the pathogen that dominates it.\n"
            "* **Separation:** the attainment margin between the weakest member of a tier and the strongest "
            "remaining pathogen, reported as a measure of how firm each tier boundary is."
        )

    with st.expander("⚖️ SSDM scenario layer and integration", expanded=False):
        st.markdown(
            "* **Scenarios:** nine registered scenarios (S1 to S9) plus the steady state ST1. Each scenario "
            "emphasizes criteria (weight multiplied by three on the elicited base) and, by default, tightens the "
            "limits of emphasized live criteria to the historical 75th percentile.\n"
            "* **Scenario probability:** $\\pi_s \\propto f_s^{\\alpha} s_s^{\\beta} h_s^{\\gamma}$ from "
            "frequency, severity and impact ratings; expert ratings blend with AI ratings through $\\lambda$.\n"
            "* **Indicators:** probability weighted robustness $R$, expected tier $\\Phi$, dispersion "
            "$\\sigma$ and stability index $SI$ place each pathogen in a decision quadrant.\n"
            "* **Integration:** agent results are combined across agents with agent weights and a second order "
            "MOSDM pass; an evidence gate withholds criteria that fail readiness checks."
        )

    # --- 4. DOMAIN AGENT REFERENCE TABLE ---
    st.markdown("**4. Analytical Domain Architecture (8 Agents)**")
    
    agent_data = [
        {"Agent": "Agent 1", "Domain": "Epidemiological Burden", "Core Metrics": "Incidence, Hospitalization Rate, ICU Rate, CFR", "Space": "Cost"},
        {"Agent": "Agent 2", "Domain": "Transmission & Susceptibility", "Core Metrics": "R0, Test Positivity, Seroprevalence, Asymptomatic %, Attack Rate", "Space": "Hybrid"},
        {"Agent": "Agent 3", "Domain": "Healthcare System Impact", "Core Metrics": "Avg Length of Stay (ALOS), ED Visit Rate, Bed Occupancy", "Space": "Cost"},
        {"Agent": "Agent 4", "Domain": "Clinical Severity & Complexity", "Core Metrics": "Days to Severe, Treatment Window, Treatment Available, AMR %, Complications", "Space": "Inverted Cost"},
        {"Agent": "Agent 5", "Domain": "Prevention & Control", "Core Metrics": "Vaccine Available, Effectiveness, Coverage, NPI Reduction, High-Risk Uptake", "Space": "Benefit"},
        {"Agent": "Agent 6", "Domain": "Equity & Vulnerable Populations", "Core Metrics": "Pediatric (<5) Hosp, Geriatric (65+) Hosp, Racial Disparity Ratio", "Space": "Cost"},
        {"Agent": "Agent 7", "Domain": "Outbreak Dynamics & Risk", "Core Metrics": "Geographic Spread Risk, Effective Rt, Outbreak Growth Probability", "Space": "Cost / Velocity"},
        {"Agent": "Agent 8", "Domain": "Social & Economic Impact", "Core Metrics": "Primary SVI Driver, Estimated Annual Direct Medical Costs ($B)", "Space": "Cost"}
    ]
    st.dataframe(pd.DataFrame(agent_data), width="stretch", hide_index=True)

    # --- 5. DATA INGESTION & CDC SURVEILLANCE PIPELINE ---
    st.markdown("**5. Automated CDC Surveillance Ingestion Pipeline**")
    st.markdown(
        "Live surveillance data is automatically paginated from official CDC Socrata open data API endpoints "
        "every Saturday at 04:00 AM UTC via GitHub Actions, populating `data/Research Data.xlsx`:"
    )

    feed_data = [
        {"Surveillance Feed": "Weekly Hospital Respiratory Data", "Endpoint ID": "ua7e-t2fy", "Network / Source": "CDC NHSN", "Target Pathogens": "COVID-19, Influenza, RSV"},
        {"Surveillance Feed": "Emergency Department Visits", "Endpoint ID": "rdmq-nq56", "Network / Source": "CDC NSSP", "Target Pathogens": "COVID-19, Influenza, RSV"},
        {"Surveillance Feed": "Viral Test Positivity", "Endpoint ID": "seuz-s2cv", "Network / Source": "CDC NREVSS", "Target Pathogens": "COVID-19, Influenza, RSV"},
        {"Surveillance Feed": "SARS-CoV-2 Wastewater Load", "Endpoint ID": "j9g8-acpt", "Network / Source": "CDC NWSS", "Target Pathogens": "SARS-CoV-2"},
        {"Surveillance Feed": "Epidemic Trends & Rt Velocity", "Endpoint ID": "5dqz-y4ea", "Network / Source": "CDC CFA", "Target Pathogens": "COVID-19, Influenza, RSV"},
        {"Surveillance Feed": "RESP-NET Hospitalization Rates", "Endpoint ID": "kvib-3txy", "Network / Source": "CDC RESP-NET", "Target Pathogens": "COVID-19, Influenza, RSV"},
        {"Surveillance Feed": "Weekly Notifiable Diseases (NNDSS)", "Endpoint ID": "x9gk-5huc", "Network / Source": "CDC NNDSS", "Target Pathogens": "Meningococcal Disease"}
    ]
    st.dataframe(pd.DataFrame(feed_data), width="stretch", hide_index=True)

    # --- 6. DATA GOVERNANCE & EVIDENCE BASE ---
    st.markdown("**6. Data Governance & Research Evidence Base**")
    st.markdown(
        "* **Zero-Imputation Policy:** Missing surveillance inputs are preserved as `NaN` rather than artificially assigned zero to prevent biased policy conclusions.\n"
        "* **Range Normalization:** Multi-center clinical report ranges (e.g., $41.0\\% - 78.0\\%$) are sanitized to arithmetic midpoints for computational consistency.\n"
        "* **Audit Trail:** Every static baseline figure is linked to a corresponding entry in the Citations index with original publication titles, CDC MMWR numbers, and direct source URLs."
    )