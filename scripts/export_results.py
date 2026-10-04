"""Headless export of every SSDM dashboard result to one workbook.

Runs the full engine outside Streamlit under the current default
configuration (lam 0.7, alpha = beta = gamma = 1, graded calibrated
attainment, National jurisdiction, theta 0.7, Dirichlet concentration 60,
500 draws, seed 7) and writes ssdm_results_export.xlsx beside the master
workbook. Each block is fault tolerant: a failing sheet is logged on the
Run_Log sheet and the remaining sheets still export.

Usage (Colab, after Cell 1 pull):
    !python /content/ssdm-dashboard/scripts/export_results.py
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

# Default configuration (mirrors the dashboard defaults)
JUR = "National"
LAM, ALPHA, BETA, GAMMA = 0.7, 1.0, 1.0, 1.0
MODE = "graded_calibrated"
THETA = 0.7
N_DRAWS, CONC, SEED = 500, 60.0, 7

LOG: list[dict] = []


def _log(sheet: str, status: str, detail: str = "") -> None:
    LOG.append({"Sheet": sheet, "Status": status, "Detail": detail[:500]})
    print(f"[{status}] {sheet} {detail[:160]}")


def _attempt(sheet: str, fn, writer) -> None:
    try:
        obj = fn()
        if isinstance(obj, pd.DataFrame):
            obj.to_excel(writer, sheet_name=sheet[:31], index=False)
        _log(sheet, "ok")
    except Exception as exc:  # noqa: BLE001
        _log(sheet, "FAILED", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


def main() -> None:
    from modules.load_data import (get_data_filepath, load_scenario_registry)
    from modules.agent_engine import AGENTS
    from modules import scenario_engine as se
    from modules import ssdm_core as sc
    from modules import sensitivity as sv

    data_path = Path(get_data_filepath()).resolve()  # follow the Drive symlink
    out_path = data_path.parent / "ssdm_results_export.xlsx"
    agent_labels = {k: sc.AGENT_LABELS[k] for k in AGENTS}

    writer = pd.ExcelWriter(out_path, engine="openpyxl")

    # Run_Config -----------------------------------------------------------
    def run_config() -> pd.DataFrame:
        rows = [
            ("exported_utc", datetime.now(timezone.utc).isoformat()),
            ("workbook", str(data_path)),
            ("jurisdiction", JUR),
            ("lambda", LAM), ("alpha", ALPHA), ("beta", BETA),
            ("gamma", GAMMA), ("attainment_mode", MODE),
            ("quadrant_theta", THETA),
            ("mc_draws", N_DRAWS), ("mc_concentration", CONC),
            ("mc_seed", SEED),
            ("note", "All judgment inputs at dashboard defaults: equal "
                     "weights, hierarchy limits, flagged assessments."),
        ]
        return pd.DataFrame(rows, columns=["Parameter", "Value"])

    _attempt("Run_Config", run_config, writer)

    # Scenario registry and probabilities ----------------------------------
    _attempt("Scenario_Registry", lambda: se.get_registry(), writer)
    _attempt("Scenario_Probabilities",
             lambda: se.probability_table(LAM, ALPHA, BETA, GAMMA), writer)

    # Agent baseline stratifications, k_w, anchors, limits ------------------
    def agent_results() -> pd.DataFrame:
        frames = []
        for k in AGENTS:
            res, prov = se.run_scenario(k, None, JUR)
            t = res.table()
            kw, theta_rows, lim_rows = {}, [], []
            for rec in res.iterations:
                for a in rec.tier:
                    kw[a] = rec.k_weighted.get(a)
                theta_rows.append(rec.thetas)
                lim_rows.append((rec.iteration, rec.limits,
                                 rec.limit_source, rec.delta))
            t.insert(0, "Agent", agent_labels[k])
            t["k_w (weighted attainment)"] = t["Alternative"].map(
                lambda a: round(kw.get(a, float("nan")), 4))
            t["Weight provenance"] = prov
            frames.append(t)
        return pd.concat(frames, ignore_index=True)

    _attempt("Agent_Stratifications", agent_results, writer)

    def anchors() -> pd.DataFrame:
        rows = []
        for k in AGENTS:
            res, _ = se.run_scenario(k, None, JUR)
            for rec in res.iterations:
                for crit, th in (rec.thetas or {}).items():
                    rows.append({"Agent": agent_labels[k],
                                 "Iteration": rec.iteration,
                                 "Criterion": crit,
                                 "theta*": th,
                                 "delta": round(rec.delta, 4)})
        return pd.DataFrame(rows)

    _attempt("Calibrated_Anchors", anchors, writer)

    def limits() -> pd.DataFrame:
        rows = []
        for k in AGENTS:
            res, _ = se.run_scenario(k, None, JUR)
            for rec in res.iterations:
                for crit, lv in rec.limits.items():
                    rows.append({"Agent": agent_labels[k],
                                 "Iteration": rec.iteration,
                                 "Criterion": crit, "Limit": lv,
                                 "Source": rec.limit_source.get(crit, "")})
        return pd.DataFrame(rows)

    _attempt("Acceptable_Limits", limits, writer)

    # Scenario tier matrices, stability, indicators, quadrants --------------
    def stacked(fn) -> pd.DataFrame:
        frames = []
        for k in AGENTS:
            df = fn(k)
            df.insert(0, "Agent", agent_labels[k])
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    _attempt("Scenario_Tier_Matrix",
             lambda: stacked(lambda k: se.tier_matrix(k, JUR).reset_index()),
             writer)
    _attempt("Stability",
             lambda: stacked(lambda k: se.stability_table(
                 k, JUR, LAM, ALPHA, BETA, GAMMA)), writer)
    _attempt("Indicators",
             lambda: stacked(lambda k: sc.indicators(
                 k, JUR, LAM, ALPHA, BETA, GAMMA, MODE)), writer)
    _attempt("Quadrants",
             lambda: stacked(lambda k: sc.quadrants(sc.indicators(
                 k, JUR, LAM, ALPHA, BETA, GAMMA, MODE), THETA)), writer)

    # Integration -----------------------------------------------------------
    def agent_weights() -> pd.DataFrame:
        w, prov = sc.get_agent_weights()
        return pd.DataFrame([{"Agent": agent_labels[k], "W_g": w[k],
                              "Provenance": prov} for k in AGENTS])

    _attempt("Agent_Weights", agent_weights, writer)
    _attempt("Second_Order_Expected",
             lambda: sc.second_order(JUR, LAM, ALPHA, BETA, GAMMA,
                                     MODE, False).reset_index(), writer)
    _attempt("Second_Order_Baseline",
             lambda: sc.second_order(JUR, LAM, ALPHA, BETA, GAMMA,
                                     MODE, True).reset_index(), writer)

    def final_integration() -> pd.DataFrame:
        frames = []
        for base_flag, name in [(False, "Expected (probability weighted)"),
                                (True, "Baseline reference")]:
            _, res, wprov = sc.cross_agent_run(JUR, LAM, ALPHA, BETA,
                                               GAMMA, MODE, base_flag)
            t = res.table()
            t.insert(0, "Matrix", name)
            t["Agent weight provenance"] = wprov
            frames.append(t)
        return pd.concat(frames, ignore_index=True)

    _attempt("Final_Integration", final_integration, writer)

    # Monte Carlo ------------------------------------------------------------
    _attempt("Monte_Carlo_Tiers",
             lambda: sv.weight_monte_carlo(JUR, LAM, ALPHA, BETA, GAMMA,
                                           MODE, N_DRAWS, CONC, SEED)[0],
             writer)

    # Evidence ledger and pipeline status -------------------------------------
    def ledger() -> pd.DataFrame:
        p = data_path.parent / "evidence_ledger.csv"
        if not p.exists():
            raise FileNotFoundError(f"no ledger at {p}")
        return pd.read_csv(p)

    _attempt("Evidence_Ledger", ledger, writer)

    def status() -> pd.DataFrame:
        p = ROOT / "data" / "pipeline_status.json"
        if not p.exists():
            p = data_path.parent / "pipeline_status.json"
        obj = json.loads(p.read_text())
        return pd.json_normalize(obj)

    _attempt("Pipeline_Status", status, writer)

    pd.DataFrame(LOG).to_excel(writer, sheet_name="Run_Log", index=False)
    writer.close()
    size = os.path.getsize(out_path) / 1024
    print(f"\nExport complete: {out_path} ({size:.0f} KB)")
    failed = [r for r in LOG if r["Status"] == "FAILED"]
    print(f"Sheets ok: {len(LOG) - len(failed)}, failed: {len(failed)}")


if __name__ == "__main__":
    main()
