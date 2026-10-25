"""
weight_manager.py

Elicitation of judgment (thesis Chapter 2, "Eliciting Judgment"; paper
Section 4.8). Linguistic terms -> trapezoidal fuzzy numbers -> aggregation
over respondents -> graded mean integration representation (GMIR) ->
normalization. Four input modes wherever judgment enters: default, direct
selection, upload (one column per respondent), questionnaire (one
respondent at a time, every respondent kept).

Aggregation rules over respondents:
  mean               componentwise arithmetic mean of the trapezoids (default);
                     by linearity of GMIR it equals the mean crisp judgment;
  minmax_geometric   minimum of the lower bounds, geometric means of the two
                     central parameters, maximum of the upper bounds
                     (Eq. ch2_aggregate_base), which keeps the full spread of
                     opinion in the support.

Scopes: an agent's baseline weights ("A1"), its weights under one scenario
("A1::S3"), and the agent level weights W_g ("agent_level::global").
Storage: modules.state (shared, persistent, provenance and author kept).
"""

from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from modules import state

IMPORTANCE_SCALE: Dict[str, Tuple[float, float, float, float]] = {
    "Very low":  (0.00, 0.00, 0.10, 0.25),
    "Low":       (0.15, 0.30, 0.40, 0.55),
    "Medium":    (0.45, 0.55, 0.65, 0.75),
    "High":      (0.65, 0.80, 0.90, 1.00),
    "Very high": (0.85, 0.95, 1.00, 1.00),
}
INTENSITY_SCALE: Dict[str, Tuple[float, float, float, float]] = {
    "Negligible": (0.0, 0.0, 0.10, 0.20),
    "Moderate":   (0.20, 0.35, 0.45, 0.60),
    "Severe":     (0.50, 0.65, 0.75, 0.90),
    "Critical":   (0.80, 0.90, 1.00, 1.00),
}
RULES = {"mean": "Componentwise mean (default)",
         "minmax_geometric": "Minimum, geometric means, maximum"}


def gmir(a: float, b: float, c: float, d: float) -> float:
    """Graded mean integration representation of a trapezoidal number."""
    return (a + 2.0 * b + 2.0 * c + d) / 6.0


def term_to_crisp(term: str, scale: Dict[str, tuple] = IMPORTANCE_SCALE) -> float:
    return gmir(*scale[term])


def aggregate_trapezoids(traps: List[Tuple[float, float, float, float]],
                         rule: str = "mean") -> Tuple[float, float, float, float]:
    t = np.array(traps, dtype=float)
    if len(t) == 0:
        raise ValueError("no judgments to aggregate")
    if rule == "minmax_geometric":
        geo = lambda col: float(np.exp(np.mean(np.log(col)))) if np.all(col > 0) else 0.0
        return (float(t[:, 0].min()), geo(t[:, 1]), geo(t[:, 2]), float(t[:, 3].max()))
    return tuple(float(v) for v in t.mean(axis=0))


def aggregate_experts(terms: List[str], scale: Dict[str, tuple] = IMPORTANCE_SCALE,
                      rule: str = "mean"):
    """One item across respondents: (aggregate trapezoid, crisp GMIR)."""
    tr = aggregate_trapezoids([scale[t] for t in terms], rule)
    return tr, gmir(*tr)


def normalize(values: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(float(v), 0.0) for v in values.values())
    n = len(values)
    if total <= 0 or n == 0:
        return {k: 1.0 / max(n, 1) for k in values}
    return {k: max(float(v), 0.0) / total for k, v in values.items()}


# ----------------------------------------------------------------------
# storage
# ----------------------------------------------------------------------

def get_entry(kind: str, key: str) -> Optional[dict]:
    """kind 'criterion' (weights of a scope) or 'limits' (expert limits)."""
    return state.get("weights" if kind == "criterion" else kind, key)


def set_entry(kind: str, key: str, values: Dict[str, float],
              provenance: str, detail: Optional[dict] = None) -> None:
    state.put("weights" if kind == "criterion" else kind, key,
              {"values": values, "detail": detail or {}}, provenance=provenance)


def get_weights(scope: str, criteria: List[str]) -> Tuple[Dict[str, float], str]:
    """Normalized weights over `criteria` with their provenance; equal weights
    flagged 'default' when the stored profile does not cover every
    criterion (a newly admitted criterion requires re elicitation, which the
    readiness gate reports)."""
    e = get_entry("criterion", scope)
    if e and set(e["values"]) >= set(criteria):
        return normalize({c: e["values"][c] for c in criteria}), e.get("provenance", "elicited")
    return {c: 1.0 / max(len(criteria), 1) for c in criteria}, "default"


def get_expert_limits(agent_key: str) -> Tuple[Dict[str, float], str]:
    e = get_entry("limits", agent_key)
    if e and e.get("values"):
        return dict(e["values"]), e.get("provenance", "expert")
    return {}, "default"


def respondents(scope: str) -> Dict[str, Dict[str, str]]:
    return state.get("respondents", scope, {}) or {}


def aggregate_respondents(scope: str, criteria: List[str],
                          scale: Dict[str, tuple] = IMPORTANCE_SCALE,
                          rule: Optional[str] = None) -> Tuple[Dict[str, float], dict]:
    rule = rule or state.setting("aggregation_rule")
    resp = respondents(scope)
    crisp, detail = {}, {}
    for c in criteria:
        terms = [r[c] for r in resp.values() if c in r]
        if not terms:
            continue
        tr, v = aggregate_experts(terms, scale, rule)
        crisp[c] = v
        detail[c] = {"terms": terms, "aggregate_trapezoid": [round(x, 4) for x in tr],
                     "crisp": round(v, 6)}
    return crisp, {"rule": rule, "respondents": list(resp), "items": detail}


# ----------------------------------------------------------------------
# elicitation UI
# ----------------------------------------------------------------------

def upload_template(criteria: List[str], scale=IMPORTANCE_SCALE) -> bytes:
    df = pd.DataFrame({"Criterion": criteria, "Respondent 1": "", "Respondent 2": "",
                       "Respondent 3": ""})
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Weights")
        pd.DataFrame({"Allowed linguistic terms": list(scale)}).to_excel(
            w, index=False, sheet_name="Scale")
    return buf.getvalue()


def parse_upload(df: pd.DataFrame, criteria: List[str], scale=IMPORTANCE_SCALE,
                 rule: str = "mean") -> Tuple[Dict[str, float], dict]:
    """Validates and converts an uploaded template. Every respondent column
    must hold either linguistic terms of the scale or nonnegative numbers,
    not a mixture; numeric columns are normalized per respondent before
    averaging, so respondents using different numeric ranges count equally."""
    df = df.set_index(df.columns[0])
    df.index = [str(i).strip() for i in df.index]
    missing = [c for c in criteria if c not in df.index]
    if missing:
        raise ValueError(f"criteria missing from the template: {missing}")
    cols = [c for c in df.columns if df.loc[criteria, c].notna().any()]
    if not cols:
        raise ValueError("no respondent column holds any value")
    lower = {k.lower(): k for k in scale}
    kinds, numeric, linguistic = {}, {}, {}
    for col in cols:
        vals = df.loc[criteria, col]
        terms = [lower.get(str(v).strip().lower()) for v in vals if pd.notna(v)]
        if all(t is not None for t in terms):
            kinds[col] = "linguistic"
            linguistic[col] = {c: lower[str(v).strip().lower()]
                               for c, v in vals.items() if pd.notna(v)}
            continue
        nums = pd.to_numeric(vals, errors="coerce")
        if nums[vals.notna()].isna().any():
            raise ValueError(f"column '{col}' mixes terms and numbers or holds "
                             f"unrecognized entries")
        if (nums.dropna() < 0).any() or not np.isfinite(nums.dropna()).all():
            raise ValueError(f"column '{col}' holds negative or nonfinite values")
        kinds[col] = "numeric"
        numeric[col] = normalize(nums.dropna().to_dict())
    if len(set(kinds.values())) > 1:
        raise ValueError("all respondent columns must use the same kind of entry "
                         "(all linguistic or all numeric)")
    crisp, detail = {}, {}
    for c in criteria:
        if linguistic:
            terms = [r[c] for r in linguistic.values() if c in r]
            if not terms:
                raise ValueError(f"no respondent rated '{c}'")
            tr, v = aggregate_experts(terms, scale, rule)
            crisp[c] = v
            detail[c] = {"terms": terms, "aggregate_trapezoid": [round(x, 4) for x in tr]}
        else:
            vs = [r[c] for r in numeric.values() if c in r]
            if not vs:
                raise ValueError(f"no respondent rated '{c}'")
            crisp[c] = float(np.mean(vs))
            detail[c] = {"values_normalized_per_respondent": [round(x, 4) for x in vs]}
    return crisp, {"respondents": cols, "kind": next(iter(kinds.values())), "rule": rule,
                   "items": detail}


def _apply(scope, values, provenance, detail):
    set_entry("criterion", scope, normalize(values), provenance, detail)
    st.session_state["_wm_flash"] = f"Applied ({provenance})."
    st.rerun()


def render_weight_elicitation(scope: str, title: str, criteria: List[str],
                              scale: Dict[str, tuple] = IMPORTANCE_SCALE) -> None:
    st.markdown(f"**Weights: {title}**")
    current, prov = get_weights(scope, criteria)
    st.caption(f"Active weights ({prov}): " +
               ", ".join(f"{c} = {w:.3f}" for c, w in current.items()))
    if st.session_state.get("_wm_flash"):
        st.success(st.session_state.pop("_wm_flash"))
    if not state.can_edit():
        st.info("Read only: sign in as editor (sidebar) to change judgment inputs.")
        return
    mode = st.radio("Input mode", ["Default (equal weights)", "Direct selection",
                                   "Upload (one column per respondent)",
                                   "Questionnaire (linguistic)"],
                    key=f"wm_mode_{scope}", horizontal=True)
    rule = state.setting("aggregation_rule")

    if mode.startswith("Default"):
        if st.button("Apply equal weights", key=f"wm_def_{scope}"):
            state.delete("weights", scope)
            st.session_state["_wm_flash"] = "Equal weights restored (flagged as default)."
            st.rerun()

    elif mode == "Direct selection":
        vals = {c: st.slider(c, 0.0, 1.0, float(current.get(c, 0.5)), 0.01,
                             key=f"wm_dir_{scope}_{c}") for c in criteria}
        if st.button("Apply (normalized automatically)", key=f"wm_dirb_{scope}"):
            _apply(scope, vals, "direct selection", {"raw": vals})

    elif mode.startswith("Upload"):
        st.download_button("Download template", data=upload_template(criteria, scale),
                           file_name=f"weights_template_{scope.replace('::', '_')}.xlsx",
                           key=f"wm_tpl_{scope}")
        up = st.file_uploader("Completed template", type=["xlsx"], key=f"wm_up_{scope}")
        if up is not None:
            try:
                crisp, detail = parse_upload(pd.read_excel(up, sheet_name=0), criteria, scale, rule)
                st.dataframe(pd.DataFrame(detail["items"]).T.astype(str), width="stretch")
                st.caption(f"{len(detail['respondents'])} respondent column(s), "
                           f"{detail['kind']} entries, rule: {RULES.get(rule, rule)}.")
                if st.button("Apply uploaded weights", key=f"wm_upb_{scope}"):
                    _apply(scope, crisp, f"upload ({len(detail['respondents'])} respondents, "
                                         f"{detail['kind']})", detail)
            except Exception as e:
                st.error(f"Template rejected: {e}")

    else:
        resp = respondents(scope)
        st.caption(f"Respondents recorded: {', '.join(resp) if resp else 'none yet'}. "
                   f"Aggregation rule: {RULES.get(rule, rule)} (set in Method settings).")
        name = st.text_input("Respondent name or code", key=f"wm_qname_{scope}")
        prev = resp.get(name, {}) if name else {}
        terms = {c: st.select_slider(c, options=list(scale),
                                     value=prev.get(c, list(scale)[len(scale) // 2]),
                                     key=f"wm_q_{scope}_{name}_{c}") for c in criteria}
        st.dataframe(pd.DataFrame({
            "Term": terms, "Trapezoid": {c: str(scale[t]) for c, t in terms.items()},
            "Crisp (GMIR)": {c: round(term_to_crisp(t, scale), 4) for c, t in terms.items()}}),
            width="stretch")
        c1, c2, c3 = st.columns(3)
        if c1.button("Save this respondent", key=f"wm_qs_{scope}", disabled=not name):
            r = respondents(scope)
            r[name] = terms
            state.put("respondents", scope, r)
            st.session_state["_wm_flash"] = f"Respondent '{name}' saved."
            st.rerun()
        if c2.button("Remove this respondent", key=f"wm_qd_{scope}", disabled=name not in resp):
            r = respondents(scope)
            r.pop(name, None)
            state.put("respondents", scope, r)
            st.rerun()
        if c3.button("Apply aggregate of all respondents", key=f"wm_qb_{scope}", disabled=not resp):
            crisp, detail = aggregate_respondents(scope, criteria, scale, rule)
            if set(crisp) >= set(criteria):
                _apply(scope, crisp, f"questionnaire ({len(resp)} respondents, {rule})", detail)
            else:
                st.error("Some criteria have no rating yet.")
        if resp:
            crisp, detail = aggregate_respondents(scope, criteria, scale, rule)
            st.dataframe(pd.DataFrame(detail["items"]).T.astype(str), width="stretch")


def render_limits_editor(agent_key: str, agent_title: str, criteria: List[str],
                         defaults: Dict[str, Tuple[float, str]]) -> None:
    """Expert acceptable limits. `defaults` maps each criterion to the limit the
    engine uses without an expert value and its level of the hierarchy
    (historical quantile or pool median at the first iteration)."""
    st.markdown(f"**Acceptable limits: {agent_title}**")
    cur, prov = get_expert_limits(agent_key)
    if st.session_state.get("_wm_flash"):
        st.success(st.session_state.pop("_wm_flash"))
    base = pd.DataFrame({
        "Subcriterion": criteria,
        "Default limit": [None if pd.isna(defaults.get(c, (np.nan, ""))[0])
                          else round(defaults[c][0], 4) for c in criteria],
        "Default source": [defaults.get(c, (np.nan, "none"))[1] for c in criteria],
        "Expert limit (blank = default)": [cur.get(c) for c in criteria],
    })
    if not state.can_edit():
        st.dataframe(base, hide_index=True, width="stretch")
        st.info("Read only: sign in as editor to set expert limits.")
        return
    edited = st.data_editor(base, hide_index=True, width="stretch",
                            key=f"lim_ed_{agent_key}",
                            disabled=["Subcriterion", "Default limit", "Default source"])
    if st.button("Apply limits", key=f"lim_b_{agent_key}"):
        vals = {}
        for _, r in edited.iterrows():
            v = r["Expert limit (blank = default)"]
            if v is not None and str(v).strip() != "" and not pd.isna(v):
                vals[r["Subcriterion"]] = float(v)
        if vals:
            state.put("limits", agent_key, {"values": vals}, provenance="expert")
        else:
            state.delete("limits", agent_key)
        st.session_state["_wm_flash"] = (f"{len(vals)} expert limit(s) active; the "
                                         "rest follow the hierarchy.")
        st.rerun()
