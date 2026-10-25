"""
controls.py

Analysis parameters and the Method settings page. A viewer may change any
setting for their own session; the editor may save the current settings
as the dashboard default for everyone (recorded with time and author).
"""

import streamlit as st

from modules import state
from modules.mosdm_core import MODES
from modules.live_subcriteria import WINDOWS
from modules.weight_manager import RULES

MODE_LABELS = {"graded_fixed": "Graded, fixed anchor (selected default)",
               "graded_calibrated": "Graded, calibrated anchors",
               "binary": "Binary (first generation rule)"}

PARAMS = {  # name: (label, min, max, step)
    "lambda": ("lambda (expert share)", 0.0, 1.0, 0.05),
    "alpha": ("alpha (frequency exponent)", 0.0, 2.0, 0.1),
    "beta": ("beta (severity exponent)", 0.0, 2.0, 0.1),
    "gamma": ("gamma (impact exponent)", 0.0, 2.0, 0.1),
    "theta_si": ("theta_SI (stability threshold)", 0.5, 0.9, 0.05),
}


def params():
    s = state.settings()
    return (float(s["lambda"]), float(s["alpha"]), float(s["beta"]), float(s["gamma"]))


def param_sliders(names=("lambda", "alpha", "beta", "gamma", "theta_si")):
    cols = st.columns(len(names))
    for col, n in zip(cols, names):
        label, lo, hi, step = PARAMS[n]
        st.session_state.setdefault(f"ctl_{n}", float(state.setting(n)))
        col.slider(label, lo, hi, step=step, key=f"ctl_{n}",
                   on_change=lambda n=n: state.set_session_setting(n, st.session_state[f"ctl_{n}"]))


def settings_page():
    st.title("Method Settings")
    st.caption("Every option of the method (thesis Chapter 2) in one place. Changes apply to "
               "your session at once; the editor can save them as the default for every viewer. "
               "Defaults are the preregistered baseline configuration.")
    s = state.settings()
    new = {}
    st.subheader("Stratification core")
    c1, c2, c3 = st.columns(3)
    new["attainment_mode"] = c1.selectbox("Attainment mode", list(MODES), index=list(MODES).index(s["attainment_mode"]),
                                          format_func=MODE_LABELS.get)
    new["theta_fixed"] = c2.number_input("Fixed anchor theta", 0.05, 0.95, float(s["theta_fixed"]), 0.05)
    lo, hi = c3.slider("Calibrated anchor bounds", 0.05, 0.95, (float(s["theta_min"]), float(s["theta_max"])), 0.05)
    new["theta_min"], new["theta_max"] = lo, hi
    c1, c2, c3 = st.columns(3)
    new["dominance_guard"] = c1.toggle("Dominance guard", bool(s["dominance_guard"]))
    new["guard_min_shared"] = int(c2.number_input("Guard: minimum shared criteria q_min", 1, 10, int(s["guard_min_shared"])))
    new["fuzzy_distance"] = c3.toggle("Fuzzy distances within tiers", bool(s["fuzzy_distance"]),
                                      help="Discretized fuzzy distance on the trapezoids of ranged evidence "
                                           "(identical to the crisp distance for interval evidence).")
    st.subheader("Limits and scenarios")
    c1, c2, c3 = st.columns(3)
    new["history_window"] = c1.selectbox("History window for limits", list(WINDOWS),
                                         index=list(WINDOWS).index(s["history_window"]), format_func=WINDOWS.get)
    new["scenario_tightening"] = c2.toggle("Scenario limit tightening", bool(s["scenario_tightening"]))
    new["tightening_quantile"] = c3.number_input("Tightening quantile", 0.5, 0.99, float(s["tightening_quantile"]), 0.05)
    c1, c2, c3, c4 = st.columns(4)
    new["emphasis_source"] = c1.selectbox("Scenario emphasis", ["data", "registry_k3"],
                                          index=["data", "registry_k3"].index(s["emphasis_source"]),
                                          format_func={"data": "Data derived multipliers (default)",
                                                       "registry_k3": "Fixed k = 3 (ablation only)"}.get)
    new["frequency_source"] = c2.selectbox("Scenario frequency", ["computed", "registry"],
                                           index=["computed", "registry"].index(s["frequency_source"]),
                                           format_func={"computed": "Recomputed weekly from the series",
                                                        "registry": "Curated registry counts"}.get)
    new["episode_gap_weeks"] = int(c3.number_input("Episode: gap tolerance (weeks)", 0, 8, int(s["episode_gap_weeks"])))
    new["episode_min_weeks"] = int(c4.number_input("Episode: minimum length (weeks)", 1, 8, int(s["episode_min_weeks"])))
    st.subheader("Judgment")
    c1, c2 = st.columns(2)
    new["aggregation_rule"] = c1.selectbox("Aggregation over respondents", list(RULES),
                                           index=list(RULES).index(s["aggregation_rule"]), format_func=RULES.get)
    new["gemini_model"] = c2.text_input("AI assistant model", s["gemini_model"])

    changed = {k: v for k, v in new.items() if v != s[k]}
    b1, b2, b3 = st.columns(3)
    if b1.button("Apply to my session", type="primary", disabled=not changed):
        for k, v in changed.items():
            state.set_session_setting(k, v)
        st.rerun()
    if b2.button("Save current settings as the default for everyone",
                 disabled=not state.can_edit()):
        for k, v in changed.items():
            state.set_session_setting(k, v)
        state.save_settings_as_default()
        st.success("Saved.")
        st.rerun()
    if b3.button("Return to the saved defaults"):
        state.clear_session_settings()
        for n in PARAMS:
            st.session_state.pop(f"ctl_{n}", None)
        st.rerun()
    if changed:
        st.info(f"{len(changed)} unapplied change(s).")
    st.caption(f"Inputs are stored in: {state.backend().name}.")
