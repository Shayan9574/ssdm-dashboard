"""Headless export of every INPUT the SSDM engine consumes, so the full
experimental programme (baseline selection, one at a time sensitivity,
factorial design, Monte Carlo, ablation, panel studies, weekly replay) can
be run offline on exactly the deployed data and code.

Writes ssdm_experiment_inputs.xlsx beside the master workbook (on Drive).
It first re-runs export_results.py so the replication target and the
inputs come from the same data snapshot.

Usage (Colab, after Cells 1 to 4):
    !cd /content/ssdm-dashboard && python scripts/export_inputs.py
    !cd /content/ssdm-dashboard && python scripts/export_inputs.py --no-ai
The AI step calls the Gemini assessor once per weighted scenario and needs
GEMINI_API_KEY (Cell 4). Its ratings are recorded, flagged, never applied.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

JUR = "National"
REPLAY_START = "2023-09-30"     # three respiratory seasons of weekly replay
LOG: list[dict] = []


def _log(block: str, status: str, detail: str = "") -> None:
    LOG.append({"Block": block, "Status": status, "Detail": detail[:500]})
    print(f"[{status}] {block} {detail[:160]}", flush=True)


def _attempt(block: str, fn):
    try:
        out = fn()
        _log(block, "ok")
        return out
    except Exception as exc:  # noqa: BLE001
        _log(block, "FAILED", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# as of replacement for the live selector (same semantics, date bounded)
# ---------------------------------------------------------------------------
AS_OF: list = [None]


def _install_as_of():
    from modules import live_connectors as lc
    original = lc.extract_completed_series

    def as_of_series(df, date_col, lag_complete_weeks=1):
        if AS_OF[0] is None or df.empty or date_col not in df.columns:
            return original(df, date_col, lag_complete_weeks)
        d = df.copy()
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.dropna(subset=[date_col])
        d = d[d[date_col] <= AS_OF[0]].sort_values(date_col).reset_index(drop=True)
        if d.empty:
            return d, None
        return d, d.iloc[max(0, len(d) - 1 - lag_complete_weeks)]

    lc.extract_completed_series = as_of_series
    return original


def _profiles(agents, core4, build) -> pd.DataFrame:
    wide = build(jurisdiction=JUR)
    wide = wide[wide["Disease Type"].isin(core4)]
    rows = []
    for k, cfg in agents.items():
        prof = cfg["profile"](wide, JUR)
        prof = prof[prof["Disease Type"].isin(core4)]
        for _, r in prof.iterrows():
            for c in cfg["criteria"]:
                rows.append({"Agent": k, "Disease": r["Disease Type"],
                             "Criterion": c,
                             "Value": pd.to_numeric(r.get(c), errors="coerce")})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--no-replay", action="store_true")
    args = ap.parse_args()

    # 1. replication target from the same snapshot
    _attempt("export_results (replication target)", lambda: subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_results.py")],
        check=True, capture_output=True, text=True))

    from modules.load_data import get_data_filepath
    from modules.agent_engine import AGENTS, CORE4
    from modules.agent_m_data import build_hybrid_decision_matrix
    from modules import scenario_engine as se
    from modules import ssdm_core as sc
    from modules import weight_manager as wm

    data_path = Path(get_data_filepath()).resolve()
    out_path = data_path.parent / "ssdm_experiment_inputs.xlsx"
    sheets: dict[str, pd.DataFrame] = {}

    # 2. meta
    def meta():
        try:
            commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse",
                                     "--short", "HEAD"], capture_output=True,
                                    text=True).stdout.strip()
        except Exception:  # noqa: BLE001
            commit = "unknown"
        return pd.DataFrame([
            ("exported_utc", datetime.now(timezone.utc).isoformat()),
            ("git_commit", commit), ("workbook", str(data_path)),
            ("jurisdiction", JUR), ("core4", "; ".join(CORE4)),
            ("integration_invert", "; ".join(sorted(sc.INTEGRATION_INVERT))),
            ("replay_start", REPLAY_START)], columns=["Key", "Value"])
    sheets["Meta"] = _attempt("Meta", meta)

    # 3. agent definitions
    def agents():
        rows = []
        for k, cfg in AGENTS.items():
            for i, c in enumerate(cfg["criteria"]):
                rows.append({"Agent": k, "Label": sc.AGENT_LABELS[k],
                             "Title": cfg["title"], "Order": i,
                             "Criterion": c, "Direction": cfg["directions"][c],
                             "Invert at integration": k in sc.INTEGRATION_INVERT})
        return pd.DataFrame(rows)
    sheets["Agents"] = _attempt("Agents", agents)

    # 4. current decision matrices
    sheets["Profiles_Current"] = _attempt(
        "Profiles_Current",
        lambda: _profiles(AGENTS, CORE4, build_hybrid_decision_matrix))

    # 5. scenario weight profiles and registry
    def scen_weights():
        rows = []
        for sid in se.scenario_ids(include_stress=True):
            for k in AGENTS:
                w, prov = se.scenario_weights(sid, k)
                for c, v in w.items():
                    rows.append({"Scenario": sid, "Agent": k, "Criterion": c,
                                 "Weight": v, "Provenance": prov})
        return pd.DataFrame(rows)
    sheets["Scenario_Weights"] = _attempt("Scenario_Weights", scen_weights)
    sheets["Registry"] = _attempt("Registry", se.get_registry)
    sheets["Probability_Defaults"] = _attempt(
        "Probability_Defaults", lambda: se.probability_table(0.7, 1, 1, 1))

    # 6. fuzzy scales
    def scales():
        rows = []
        for name, scale in [("intensity", se.INTENSITY_SCALE),
                            ("importance", wm.IMPORTANCE_SCALE)]:
            for term, tfn in scale.items():
                rows.append({"Scale": name, "Term": term, "a": tfn[0],
                             "b": tfn[1], "c": tfn[2], "d": tfn[3],
                             "GMIR": wm.gmir(*tfn)})
        return pd.DataFrame(rows)
    sheets["Scales"] = _attempt("Scales", scales)

    # 7. AI assessor ratings (recorded, flagged, never applied)
    if not args.no_ai:
        def ai():
            rows = []
            reg = se.get_registry()
            for _, r in reg[reg["weighted"]].iterrows():
                try:
                    sev, imp, why = se.gemini_assess(r)
                    rows.append({"Scenario": r["Scenario ID"], "Severity": sev,
                                 "Impact": imp, "Justification": why,
                                 "Model": se.DEFAULT_GEMINI_MODEL,
                                 "Status": "ok"})
                except Exception as exc:  # noqa: BLE001
                    rows.append({"Scenario": r["Scenario ID"],
                                 "Status": f"{type(exc).__name__}: {exc}"[:300]})
            return pd.DataFrame(rows)
        sheets["AI_Assessments"] = _attempt("AI_Assessments", ai)

    # 8. weekly replay of the decision matrices
    if not args.no_replay:
        def replay():
            _install_as_of()
            end = pd.Timestamp(datetime.now(timezone.utc).date())
            dates = pd.date_range(REPLAY_START, end, freq="7D")
            frames = []
            for i, d in enumerate(dates):
                AS_OF[0] = d
                try:
                    p = _profiles(AGENTS, CORE4, build_hybrid_decision_matrix)
                    p.insert(0, "As_of", d.date().isoformat())
                    frames.append(p)
                except Exception as exc:  # noqa: BLE001
                    _log(f"replay {d.date()}", "FAILED", str(exc))
                if i % 20 == 0:
                    print(f"  replay {i + 1}/{len(dates)} {d.date()}", flush=True)
            AS_OF[0] = None
            return pd.concat(frames, ignore_index=True)
        sheets["Replay_Profiles"] = _attempt("Replay_Profiles", replay)

    sheets["Run_Log"] = pd.DataFrame(LOG)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            if isinstance(df, pd.DataFrame):
                df.to_excel(xw, sheet_name=name[:31], index=False)
    failed = [r for r in LOG if r["Status"] == "FAILED"]
    print(f"\nInputs export complete: {out_path}")
    print(f"Blocks ok: {len(LOG) - len(failed)}, failed: {len(failed)}")


if __name__ == "__main__":
    main()
