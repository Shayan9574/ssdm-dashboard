"""
readiness.py

The readiness gate of Section 4.4. Before any stratification is displayed,
three blocking conditions are verified:
  R1 data typed and harmonized (percent units, numeric active criteria),
  R2 scenario registry current (valid episode counts and windows),
  R3 weights complete for every ACTIVE subcriterion (an elicited profile
     that does not cover a newly admitted live criterion must be
     re-elicited; the default mode always passes and is flagged).
Two further checks are informational: R4 evidence currency (staleness
window) and R5 the coverage report (active and not admitted criteria).

render_gate() shows the status and, on any blocking failure, stops the
result pages with the refine and re-run action (clear caches, re-read the
workbook, re-evaluate). Pages where refinement happens (Evidence & Data,
Elicitation Studio, Methods) are never blocked.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

STALE_DAYS = 7


def _check(name, ok, detail, blocking=True, warn=False):
    status = "pass" if ok else ("warn" if warn else "fail")
    return {"Check": name, "Status": status, "Blocking": bool(blocking and not ok and not warn),
            "Detail": detail}


def evaluate(jurisdiction: str = "National") -> List[Dict]:
    from modules.load_data import load_wide, percent_unit_violations, harmonization_log
    from modules.agent_engine import AGENTS, CORE4, active_criteria
    from modules.agent_m_data import get_active_decision_matrix
    from modules import weight_manager as wm
    from modules import live_subcriteria as ls
    from modules import scenario_engine as se

    out = []
    # R1 data
    try:
        wide = load_wide()
        viol = percent_unit_violations(wide)
        n_h = len(harmonization_log())
        out.append(_check("R1 Data typed and harmonized", viol.empty,
                          f"{n_h} percent cells harmonized from stored fractions; "
                          f"{len(viol)} values outside 0 to 100"))
    except Exception as e:  # noqa: BLE001
        out.append(_check("R1 Data typed and harmonized", False, f"error: {e}"))
        wide = None

    # R5 coverage (also needed for R3)
    active: Dict[str, List[str]] = {}
    try:
        w = get_active_decision_matrix(jurisdiction=jurisdiction)
        w = w[w["Disease Type"].isin(CORE4)]
        inactive = []
        for k, cfg in AGENTS.items():
            prof = cfg["profile"](w, jurisdiction)
            prof = prof[prof["Disease Type"].isin(CORE4)]
            active[k] = active_criteria(prof, k)
            inactive += [f"{k}: {c}" for c in cfg["criteria"] if c not in active[k]]
        n_act = sum(len(v) for v in active.values())
        n_live = sum(1 for k in active for c in active[k] if c in ls.criteria_for(k))
        detail = (f"{n_act} active subcriteria ({n_live} live); inactive under the "
                  f"coverage rule: {', '.join(inactive) if inactive else 'none'}; "
                  f"not admitted by design: "
                  + "; ".join(f"{a}: {n} ({why})" for a, n, why in ls.NOT_ADMITTED))
        thin = [k for k, v in active.items() if len(v) < 2]
        out.append(_check("R5 Coverage report", not thin,
                          detail + (f"; single criterion agents: {', '.join(thin)}" if thin else ""),
                          blocking=False, warn=bool(thin)))
    except Exception as e:  # noqa: BLE001
        out.append(_check("R5 Coverage report", False, f"error: {e}", blocking=False, warn=True))

    # R2 registry
    try:
        reg = se.get_registry()
        bad = []
        for _, r in reg[reg["weighted"]].iterrows():
            n = pd.to_numeric(r["n_s"], errors="coerce")
            y = pd.to_numeric(r["Y_s (years observable)"], errors="coerce")
            if not (n == n and y == y and n >= 1 and y > 0):
                bad.append(r["Scenario ID"])
        out.append(_check("R2 Scenario registry current", not bad,
                          f"{int(reg['weighted'].sum())} weighted scenarios; invalid: "
                          f"{', '.join(bad) if bad else 'none'}"))
    except Exception as e:  # noqa: BLE001
        out.append(_check("R2 Scenario registry current", False, f"error: {e}"))

    # R3 weights
    try:
        gaps, defaults = [], []
        for k in AGENTS:
            e = wm.get_entry("criterion", k)
            if not e:
                defaults.append(k)
                continue
            missing = [c for c in active.get(k, []) if c not in e["values"]]
            if missing:
                gaps.append(f"{k} (re-elicit: {', '.join(missing)})")
        detail = ("incomplete: " + "; ".join(gaps) + ". ") if gaps else ""
        detail += ("default mode (equal weights, flagged) for: "
                   + (", ".join(defaults) if defaults else "none"))
        out.append(_check("R3 Weights complete for active subcriteria", not gaps, detail))
    except Exception as e:  # noqa: BLE001
        out.append(_check("R3 Weights complete for active subcriteria", False, f"error: {e}"))

    # R4 currency
    try:
        from modules.load_data import get_data_filepath
        p = Path(get_data_filepath()).resolve().parent / "pipeline_status.json"
        if not p.exists():
            p = Path(__file__).resolve().parent.parent / "data" / "pipeline_status.json"
        ts = json.loads(p.read_text())["last_sync_timestamp"]
        age = (datetime.now(timezone.utc) - datetime.strptime(ts, "%Y-%m-%d %H:%M UTC")
               .replace(tzinfo=timezone.utc)).total_seconds() / 86400
        out.append(_check("R4 Evidence currency", age <= STALE_DAYS,
                          f"last synchronization {ts} ({age:.1f} days)",
                          blocking=False, warn=age > STALE_DAYS))
    except Exception as e:  # noqa: BLE001
        out.append(_check("R4 Evidence currency", False, f"status unavailable: {e}",
                          blocking=False, warn=True))
    return out


def render_gate(jurisdiction: str, page: str) -> None:
    """Sidebar status plus a blocking stop on result pages."""
    try:
        checks = evaluate(jurisdiction)
    except Exception as e:  # noqa: BLE001  (the gate must never break the app)
        st.sidebar.warning(f"Readiness gate unavailable: {e}")
        return
    blocking = [c for c in checks if c["Blocking"]]
    warns = [c for c in checks if c["Status"] == "warn"]
    with st.sidebar:
        if blocking:
            st.error(f"Readiness: {len(blocking)} blocking check(s) failed")
        elif warns:
            st.warning(f"Readiness: ready, {len(warns)} note(s)")
        else:
            st.success("Readiness: all checks passed")
        with st.expander("Readiness gate"):
            st.dataframe(pd.DataFrame(checks), hide_index=True, width="stretch")
    if blocking and page not in {"Evidence & Data", "Elicitation Studio", "Methods & About"}:
        st.error("The readiness gate stopped this view (Section 4.4). Resolve the "
                 "checks below, then refine and re-run.")
        st.dataframe(pd.DataFrame(blocking), hide_index=True, width="stretch")
        if st.button("Refine and re-run", type="primary"):
            st.cache_data.clear()
            st.rerun()
        st.stop()
