"""
readiness.py

The readiness gate (paper Section 4.4). Before results are shown:
  R1 data typed and harmonized (percent units within 0 to 100)      blocking
  R2 at least one weighted scenario with an episode in its window   blocking
     (a scenario without one is weighted zero and listed)              note
  R3 weights complete for every active subcriterion                 blocking
     (an elicited profile that misses a newly admitted criterion must be
     re elicited; the default mode passes and is flagged)
  R4 evidence currency: every feed present, the latest NHSN week no older
     than STALE_DAYS, and the live values that are stale               note
  R5 coverage: active, inactive and not admitted criteria                note
Pages where refinement happens are never blocked.
"""
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
import streamlit as st

from modules import state

STALE_DAYS = 10   # a weekly feed plus a three day publication margin


def _check(name, ok, detail, blocking=True, warn=False):
    status = "pass" if ok else ("note" if warn else "fail")
    return {"Check": name, "Status": status, "Blocking": bool(blocking and not ok and not warn),
            "Detail": detail}


@state.cached
def evaluate(jurisdiction: str = "National") -> List[Dict]:
    from modules.load_data import load_wide, percent_unit_violations, harmonization_log
    from modules.agent_engine import AGENTS, CORE4, active_criteria, profile
    from modules import weight_manager as wm
    from modules import live_subcriteria as ls
    from modules import scenario_engine as se
    from modules import feeds
    from modules.agents import agent7_outbreak

    out = []
    try:
        wide = load_wide()
        viol = percent_unit_violations(wide)
        out.append(_check("R1 Data typed and harmonized", viol.empty,
                          f"{len(harmonization_log())} percent cells harmonized from stored "
                          f"fractions; {len(viol)} values outside 0 to 100"))
    except Exception as e:  # noqa: BLE001
        out.append(_check("R1 Data typed and harmonized", False, f"error: {e}"))

    active: Dict[str, List[str]] = {}
    try:
        inactive = []
        for k, cfg in AGENTS.items():
            prof = profile(k, jurisdiction, tuple(CORE4))
            active[k] = active_criteria(prof, k)
            inactive += [f"{k}: {c}" for c in cfg["criteria"] if c not in active[k]]
        n_act = sum(len(v) for v in active.values())
        thin = [k for k, v in active.items() if len(v) < 2]
        out.append(_check("R5 Coverage", not thin,
                          f"{n_act} active subcriteria; inactive under the coverage rule: "
                          f"{', '.join(inactive) if inactive else 'none'}; not admitted by design: "
                          + "; ".join(f"{a}: {n}" for a, n, _ in ls.NOT_ADMITTED)
                          + (f"; single criterion agents: {', '.join(thin)}" if thin else ""),
                          blocking=False, warn=bool(thin)))
    except Exception as e:  # noqa: BLE001
        out.append(_check("R5 Coverage", False, f"error: {e}", blocking=False, warn=True))

    try:
        adm = se.admission_table()
        w = adm[adm["weighted"]]
        valid = w[(w["n_s"] >= 1) & (w["Y_s"] > 0)]
        none = [s for s in w["Scenario ID"] if s not in set(valid["Scenario ID"])]
        n_comp = int((adm["Source"] == "computed from the weekly series").sum())
        out.append(_check("R2 Scenario registry and admission", len(valid) > 0 and not none,
                          f"{len(w)} weighted scenarios ({n_comp} recomputed from the weekly series); "
                          f"without an episode in their window, so weighted zero: {', '.join(none) or 'none'}",
                          blocking=len(valid) == 0, warn=len(valid) > 0))
    except Exception as e:  # noqa: BLE001
        out.append(_check("R2 Scenario registry and admission", False, f"error: {e}"))

    try:
        gaps, defaults = [], []
        for k in AGENTS:
            e = wm.get_entry("criterion", k)
            if not e:
                defaults.append(k)
                continue
            missing = [c for c in active.get(k, []) if c not in e["values"]]
            if missing:
                gaps.append(f"{k} (re elicit: {', '.join(missing)})")
        out.append(_check("R3 Weights complete for active subcriteria", not gaps,
                          ("incomplete: " + "; ".join(gaps) + ". " if gaps else "")
                          + "default (equal weights, flagged) for: " + (", ".join(defaults) or "none")))
    except Exception as e:  # noqa: BLE001
        out.append(_check("R3 Weights complete for active subcriteria", False, f"error: {e}"))

    try:
        stt = feeds.data_status()
        missing = stt[stt["Source"] == "missing"]["Feed"].tolist()
        ref = ls.reference_week(jurisdiction)
        age = (datetime.now(timezone.utc).replace(tzinfo=None) - ref).days if ref is not None else None
        stale_vals = pd.concat([ls.freshness_report(jurisdiction), pd.DataFrame(agent7_outbreak.rt_current(jurisdiction)[1])])
        ok = not missing and age is not None and age <= STALE_DAYS
        out.append(_check("R4 Evidence currency", ok,
                          f"reference week {str(ref)[:10] if ref is not None else 'none'}"
                          f" ({age if age is not None else '?'} days ago); missing feeds: "
                          f"{', '.join(missing) or 'none'}; live values treated as missing because "
                          f"stale or out of scope: {len(stale_vals)}",
                          blocking=False, warn=not ok))
    except Exception as e:  # noqa: BLE001
        out.append(_check("R4 Evidence currency", False, f"status unavailable: {e}",
                          blocking=False, warn=True))
    return out


def render_gate(jurisdiction: str, page: str, blocking: bool = True) -> None:
    try:
        checks = evaluate(jurisdiction)
    except Exception as e:  # noqa: BLE001
        st.sidebar.warning(f"Readiness gate unavailable: {e}")
        return
    blk = [c for c in checks if c["Blocking"]]
    notes = [c for c in checks if c["Status"] == "note"]
    with st.sidebar:
        if blk:
            st.error(f"Readiness: {len(blk)} blocking check(s) failed")
        elif notes:
            st.warning(f"Readiness: ready, {len(notes)} note(s)")
        else:
            st.success("Readiness: all checks passed")
        with st.expander("Readiness gate"):
            st.dataframe(pd.DataFrame(checks), hide_index=True, width="stretch")
    if blk and blocking:
        st.error("The readiness gate stopped this view. Resolve the checks below in the "
                 "Elicitation Studio or Method Settings, then re run.")
        st.dataframe(pd.DataFrame(blk), hide_index=True, width="stretch")
        st.stop()
