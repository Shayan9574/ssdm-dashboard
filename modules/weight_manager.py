"""
weight_manager.py

Weight Manager and elicitation interface (methods document, Section 7).

Fuzzy chain: linguistic terms -> trapezoidal fuzzy numbers -> graded mean
integration representation (GMIR) -> normalization. Four input modes
everywhere judgment enters (Section 7.3): default, direct selection,
expert upload, in-app questionnaire. Multiple experts aggregate by the
componentwise mean of their trapezoidal numbers before defuzzification;
by linearity of GMIR this equals the mean of individual crisp values.

Storage: st.session_state["wm_store"], keyed by scope:
  ("criterion", agent_key)  -> weights for that agent's subcriteria
  ("limits",    agent_key)  -> expert acceptable limits (subset of criteria)
  ("agent_level", "global") -> importance weights of the eight agents
Each entry: {"values": {...}, "provenance": mode, "detail": {...}}
"""

from io import BytesIO
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# five term importance scale (Section 7.1); anchors configurable
IMPORTANCE_SCALE: Dict[str, Tuple[float, float, float, float]] = {
    "Very low":  (0.00, 0.00, 0.10, 0.25),
    "Low":       (0.15, 0.30, 0.40, 0.55),
    "Medium":    (0.45, 0.55, 0.65, 0.75),
    "High":      (0.65, 0.80, 0.90, 1.00),
    "Very high": (0.85, 0.95, 1.00, 1.00),
}

# four term intensity scale (Section 11.3), used later for scenarios
INTENSITY_SCALE: Dict[str, Tuple[float, float, float, float]] = {
    "Negligible": (0.0, 0.0, 0.10, 0.20),
    "Moderate":   (0.20, 0.35, 0.45, 0.60),
    "Severe":     (0.50, 0.65, 0.75, 0.90),
    "Critical":   (0.80, 0.90, 1.00, 1.00),
}


def gmir(a: float, b: float, c: float, d: float) -> float:
    """Graded mean integration representation of a trapezoidal number."""
    return (a + 2.0 * b + 2.0 * c + d) / 6.0


def term_to_crisp(term: str, scale: Dict[str, tuple] = IMPORTANCE_SCALE) -> float:
    return gmir(*scale[term])


def aggregate_experts(term_lists: List[List[str]],
                      scale: Dict[str, tuple] = IMPORTANCE_SCALE
                      ) -> Tuple[Tuple[float, float, float, float], float]:
    """Componentwise mean of the experts' trapezoidal numbers, then GMIR.
    term_lists: one list of terms per criterion position is NOT expected
    here; this aggregates one criterion across experts."""
    traps = np.array([scale[t] for t in term_lists], dtype=float)
    mean_trap = tuple(traps.mean(axis=0))
    return mean_trap, gmir(*mean_trap)


def normalize(values: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in values.values())
    n = len(values)
    if total <= 0:
        return {k: 1.0 / n for k in values}
    return {k: max(v, 0.0) / total for k, v in values.items()}


# ----------------------------------------------------------------------
# storage
# ----------------------------------------------------------------------

def _store() -> dict:
    if "wm_store" not in st.session_state:
        st.session_state["wm_store"] = {}
    return st.session_state["wm_store"]


def get_entry(kind: str, key: str) -> Optional[dict]:
    return _store().get((kind, key))


def set_entry(kind: str, key: str, values: Dict[str, float],
              provenance: str, detail: Optional[dict] = None) -> None:
    _store()[(kind, key)] = {"values": values, "provenance": provenance,
                             "detail": detail or {}}


def get_weights(agent_key: str, criteria: List[str]) -> Tuple[Dict[str, float], str]:
    """Returns (normalized weights, provenance) for an agent; equal weights
    with provenance 'default' when nothing has been elicited."""
    e = get_entry("criterion", agent_key)
    if e and set(e["values"]) >= set(criteria):
        return normalize({c: e["values"][c] for c in criteria}), e["provenance"]
    return {c: 1.0 / len(criteria) for c in criteria}, "default"


def get_expert_limits(agent_key: str) -> Tuple[Dict[str, float], str]:
    e = get_entry("limits", agent_key)
    if e:
        return dict(e["values"]), e["provenance"]
    return {}, "default"


# ----------------------------------------------------------------------
# elicitation UI (one widget block per agent scope)
# ----------------------------------------------------------------------

def upload_template(criteria: List[str]) -> bytes:
    """Excel template: one row per criterion, one column per expert."""
    df = pd.DataFrame({
        "Criterion": criteria,
        "Expert 1": ["" for _ in criteria],
        "Expert 2": ["" for _ in criteria],
        "Expert 3": ["" for _ in criteria],
    })
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Weights")
        pd.DataFrame({"Allowed linguistic terms": list(IMPORTANCE_SCALE)}
                     ).to_excel(w, index=False, sheet_name="Scale")
    return buf.getvalue()


def render_weight_elicitation(agent_key: str, agent_title: str,
                              criteria: List[str]) -> None:
    """The four mode elicitation block for one agent's criterion weights."""
    st.markdown(f"**Subcriterion weights: {agent_title}**")
    current, prov = get_weights(agent_key, criteria)
    st.caption(f"Active weights ({prov}): " +
               ", ".join(f"{c} = {w:.3f}" for c, w in current.items()))

    mode = st.radio(
        "Input mode", ["Default (equal weights)", "Direct selection",
                       "Expert upload", "Questionnaire (linguistic)"],
        key=f"wm_mode_{agent_key}", horizontal=True,
        help="Section 7.3: default, direct selection, expert upload, or the in-app linguistic questionnaire.")

    if mode == "Default (equal weights)":
        if st.button("Apply equal weights", key=f"wm_def_{agent_key}"):
            set_entry("criterion", agent_key,
                      {c: 1.0 / len(criteria) for c in criteria}, "default")
            st.success("Equal weights applied (flagged as default).")

    elif mode == "Direct selection":
        vals = {}
        for c in criteria:
            vals[c] = st.slider(c, 0.0, 1.0, float(current.get(c, 0.5)),
                                0.01, key=f"wm_dir_{agent_key}_{c}")
        if st.button("Apply (normalized automatically)", key=f"wm_dirb_{agent_key}"):
            set_entry("criterion", agent_key, normalize(vals), "direct selection")
            st.success("Weights applied: " + ", ".join(
                f"{c} = {v:.3f}" for c, v in normalize(vals).items()))

    elif mode == "Expert upload":
        st.download_button("Download Excel template",
                           data=upload_template(criteria),
                           file_name=f"weights_template_{agent_key}.xlsx",
                           key=f"wm_tpl_{agent_key}")
        up = st.file_uploader("Upload the completed template (crisp values or linguistic terms)",
                              type=["xlsx"], key=f"wm_up_{agent_key}")
        if up is not None:
            try:
                df = pd.read_excel(up, sheet_name=0)
                df = df.set_index(df.columns[0])
                per_crit, detail = {}, {}
                for c in criteria:
                    row = df.loc[c].dropna()
                    crisps, traps = [], []
                    for v in row:
                        s = str(v).strip()
                        if s.title() in IMPORTANCE_SCALE:
                            traps.append(IMPORTANCE_SCALE[s.title()])
                            crisps.append(gmir(*IMPORTANCE_SCALE[s.title()]))
                        else:
                            crisps.append(float(s))
                    if traps and len(traps) == len(crisps):
                        mean_trap = tuple(np.mean(traps, axis=0))
                        per_crit[c] = gmir(*mean_trap)
                        detail[c] = {"terms": list(row), "mean_trapezoid": mean_trap,
                                     "crisp": per_crit[c]}
                    else:
                        per_crit[c] = float(np.mean(crisps))
                        detail[c] = {"values": list(row), "crisp": per_crit[c]}
                if st.button("Apply uploaded weights", key=f"wm_upb_{agent_key}"):
                    set_entry("criterion", agent_key, normalize(per_crit),
                              "expert upload", detail)
                    st.success(f"Applied from {len(df.columns)} expert column(s), "
                               "aggregated by componentwise mean before GMIR.")
                st.dataframe(pd.DataFrame(detail).T, width="stretch")
            except Exception as e:
                st.error(f"Template could not be read: {e}")

    else:  # questionnaire
        st.caption("Rate the importance of each subcriterion. The conversion "
                   "chain (term, trapezoidal fuzzy number, crisp GMIR value) "
                   "is shown for transparency.")
        terms = {}
        for c in criteria:
            terms[c] = st.select_slider(c, options=list(IMPORTANCE_SCALE),
                                        value="Medium",
                                        key=f"wm_q_{agent_key}_{c}")
        chain = pd.DataFrame({
            "Term": {c: t for c, t in terms.items()},
            "Trapezoidal (a, b, c, d)": {c: str(IMPORTANCE_SCALE[t]) for c, t in terms.items()},
            "Crisp (GMIR)": {c: round(term_to_crisp(t), 4) for c, t in terms.items()},
        })
        st.dataframe(chain, width="stretch")
        if st.button("Apply questionnaire weights", key=f"wm_qb_{agent_key}"):
            crisp = {c: term_to_crisp(t) for c, t in terms.items()}
            set_entry("criterion", agent_key, normalize(crisp),
                      "questionnaire",
                      {c: {"term": t, "trapezoid": IMPORTANCE_SCALE[t],
                           "crisp": crisp[c]} for c, t in terms.items()})
            st.success("Questionnaire weights applied: " + ", ".join(
                f"{c} = {v:.3f}" for c, v in normalize(crisp).items()))


def render_limits_editor(agent_key: str, agent_title: str,
                         profile: pd.DataFrame, criteria: List[str],
                         medians: Dict[str, float]) -> None:
    """Expert acceptable limits (Section 9.3): median defaults prefilled;
    an expert value replaces the median entirely and stays fixed."""
    st.markdown(f"**Acceptable limits: {agent_title}**")
    cur, prov = get_expert_limits(agent_key)
    base = pd.DataFrame({
        "Subcriterion": criteria,
        "Median (default)": [round(medians.get(c, float("nan")), 4) for c in criteria],
        "Expert limit (blank = use median)": [cur.get(c, None) for c in criteria],
    })
    edited = st.data_editor(base, hide_index=True, width="stretch",
                            key=f"lim_ed_{agent_key}",
                            disabled=["Subcriterion", "Median (default)"])
    if st.button("Apply limits", key=f"lim_b_{agent_key}"):
        vals = {}
        for _, r in edited.iterrows():
            v = r["Expert limit (blank = use median)"]
            if v is not None and str(v).strip() != "" and not pd.isna(v):
                vals[r["Subcriterion"]] = float(v)
        set_entry("limits", agent_key, vals,
                  "expert" if vals else "default")
        st.success(f"{len(vals)} expert limit(s) active; the rest use the "
                   "median default, recomputed each iteration.")
