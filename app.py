import streamlit as st

st.set_page_config(page_title="SSDM Public Health Dashboard",
                   page_icon="🩺", layout="wide")

from modules import ui
from modules.status_banner import render_status_banner

ui.apply_theme()

# global parameters, shared by every page
for k, v in {"g_lam": 0.7, "g_alpha": 1.0, "g_beta": 1.0, "g_gamma": 1.0,
             "g_theta": 0.7, "attainment_mode": "graded_calibrated",
             "use_evidence_overlay": True,
             "scenario_limit_tightening": False,
             "g_model": "gemini-2.5-flash"}.items():
    st.session_state.setdefault(k, v)

with st.sidebar:
    st.markdown("### SSDM Dashboard")
    st.caption("Infectious Disease Prioritization")
    jurisdiction = st.selectbox(
        "Surveillance scope", ["National", "MI"],
        format_func=lambda x: "US National" if x == "National"
        else "Michigan (MI)",
        key="global_jurisdiction")
    st.markdown("**Analysis**")
    page = st.radio("Navigate", [
        "Overview",
        "Domain Agents",
        "Scenarios & Probability",
        "SSDM Integration",
        "Sensitivity & Simulation",
        "Evidence & Data",
        "Elicitation Studio",
        "Methods & About",
    ], label_visibility="collapsed")
    st.toggle("Scenario limit tightening", key="scenario_limit_tightening",
              help="Section 4.6: under each scenario, its emphasized live "
                   "subcriteria use the upper quartile of their own history "
                   "as the acceptable limit. Off by default.")
    st.divider()
    st.caption("Second generation MOSDM engine; every result carries its "
               "audit trail. Weekly CDC feeds, launch time freshness.")

from modules import pages_e
from modules import readiness

readiness.render_gate(jurisdiction, page)

if page == "Overview":
    render_status_banner()
    pages_e.overview(jurisdiction)

elif page == "Domain Agents":
    from modules.agent_engine import AGENTS
    from modules.agent_view import render_agent
    ak = st.selectbox("Domain agent", list(AGENTS),
                      format_func=lambda k: AGENTS[k]["title"],
                      key="nav_agent")
    render_agent(ak, jurisdiction=jurisdiction)

elif page == "Scenarios & Probability":
    from modules import scenario_view
    scenario_view.render(jurisdiction=jurisdiction)

elif page == "SSDM Integration":
    from modules import ssdm_view
    ssdm_view.render(jurisdiction=jurisdiction)

elif page == "Sensitivity & Simulation":
    pages_e.sensitivity_page(jurisdiction)

elif page == "Evidence & Data":
    pages_e.evidence_data(jurisdiction)

elif page == "Elicitation Studio":
    pages_e.elicitation_studio(jurisdiction)

elif page == "Methods & About":
    from modules import about
    about.render()
