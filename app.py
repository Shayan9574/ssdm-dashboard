import os

import streamlit as st

st.set_page_config(page_title="SSDM Public Health Dashboard", page_icon="🩺", layout="wide")

# deployment secrets into the environment (Gemini key, data repository)
for _k in ("GEMINI_API_KEY", "GITHUB_TOKEN", "SSDM_DATA_REPO", "SSDM_DATA_BRANCH"):
    try:
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        pass

from modules import ui, state, controls  # noqa: E402

ui.apply_theme()

PAGES = ["Overview", "Domain Agents", "Scenarios & Probability", "SSDM Integration",
         "Sensitivity & Simulation", "Method Trace", "External Evidence & Data",
         "Elicitation Studio", "Method Settings", "Methods & About"]
REFINE_PAGES = {"External Evidence & Data", "Elicitation Studio", "Method Settings", "Methods & About"}

with st.sidebar:
    st.markdown("### SSDM Dashboard")
    st.caption("Infectious Disease Prioritization")
    jurisdiction = st.selectbox("Surveillance scope", ["National", "MI"],
                                format_func=lambda x: "US National" if x == "National" else "Michigan (MI)",
                                key="global_jurisdiction")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed", key="nav_page")
    st.divider()
    if state.password_configured():
        if state.can_edit():
            st.success(f"Editor: {state.editor_name()}")
            if st.button("Sign out"):
                state.logout()
                st.rerun()
        else:
            with st.expander("Editor sign in"):
                nm = st.text_input("Name", key="login_name")
                pw = st.text_input("Password", type="password", key="login_pw")
                if st.button("Sign in"):
                    if state.login(pw, nm):
                        st.rerun()
                    else:
                        st.error("Incorrect password.")
    if os.environ.get("SSDM_STORE_ERROR"):
        st.warning(os.environ["SSDM_STORE_ERROR"])
    st.caption(f"Inputs stored in: {state.backend().name}")

from modules import pages_e, readiness  # noqa: E402

readiness.render_gate(jurisdiction, page, blocking=page not in REFINE_PAGES)

if page == "Overview":
    pages_e.overview(jurisdiction)
elif page == "Domain Agents":
    from modules.agent_engine import AGENTS
    from modules.agent_view import render_agent
    ak = st.selectbox("Domain agent", list(AGENTS), format_func=lambda k: AGENTS[k]["title"],
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
elif page == "Method Trace":
    from modules import method_trace
    method_trace.render(jurisdiction)
elif page == "External Evidence & Data":
    pages_e.evidence_data(jurisdiction)
elif page == "Elicitation Studio":
    pages_e.elicitation_studio(jurisdiction)
elif page == "Method Settings":
    controls.settings_page()
elif page == "Methods & About":
    from modules import about
    about.render()
