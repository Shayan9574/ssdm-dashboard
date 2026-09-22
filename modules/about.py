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
        "Standard public health prioritization often suffers from **premature score flattening**—collapsing "
        "diverse epidemiological dimensions (such as high annual incidence versus low-incidence, high-lethality threats) "
        "into an arbitrary single weighted average. The **SSDM Public Health Dashboard** resolves this by pairing "
        "**Multi-Objective Spatial Decision Making (MOSDM)** Pareto dominance tiering with **Scenario-Based Spatial "
        "Decision Making (SSDM)** dynamic tie-breaking."
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
    
    with st.expander("📐 Multi-Objective Spatial Decision Making (MOSDM) — Pareto Dominance Tiering", expanded=True):
        st.markdown(
            "In multi-objective optimization, pathogen alternative $A$ **dominates** pathogen $B$ in Cost Space if "
            "$A$ exhibits greater or equal burden across all criteria, and is strictly worse in at least one:\n\n"
            "$$\\forall k \\in \\{1, \\dots, m\\}, \\quad f_k(A) \\ge f_k(B) \\quad \\land \\quad \\exists k : f_k(A) > f_k(B)$$\n\n"
            "* **Tier 1 (Pareto Frontier):** The set of non-dominated pathogens. No single pathogen overshadows them across all dimensions.\n"
            "* **Iterative Peeling:** Once Tier 1 is isolated, it is removed from the candidate pool. Non-dominated alternatives among the remaining pathogens form Tier 2, repeating until all diseases are classified.\n"
            "* **Optimization Spaces:**\n"
            "  * **Cost Space (Agents 1, 3, 6, 7, 8):** Higher values represent greater risk/burden.\n"
            "  * **Benefit Space (Agent 5):** Higher values represent superior prevention capacity ($A \\ge B$ is preferred).\n"
            "  * **Inverted Cost Space (Agent 4):** Shorter clinical windows are inverted ($-x$) so higher transformed values represent greater clinical urgency."
        )

    with st.expander("⚖️ Scenario-Based Spatial Decision Making (SSDM) — Dynamic Tie-Breaking", expanded=False):
        st.markdown(
            "Within any given Pareto tier, pathogens cannot be distinguished by strict dominance alone. "
            "The system uses **SSDM linear weighted summation** across min-max normalized criteria to break intra-tier ties:\n\n"
            "$$\\bar{x}_{i,k} = \\frac{x_{i,k} - \\min(x_k)}{\\max(x_k) - \\min(x_k)}, \\qquad S_i = \\sum_{k=1}^m w_k \\bar{x}_{i,k}$$\n\n"
            "* **Scenario Weights ($w_k$):** Dynamic policy weights calibrated to 12 operational public health emergencies (e.g., *Winter Respiratory Surge*, *AMR Crisis*, *Vaccine Supply Shortage*).\n"
            "* **Final Hierarchy:** Pathogens are sorted primarily by **Pareto Tier** ($1 \\rightarrow 2 \\rightarrow 3$) and secondarily by **Weighted Score** ($S_i$)."
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