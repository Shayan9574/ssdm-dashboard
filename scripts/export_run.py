"""
export_run.py

Headless export of one complete run of the method: every input, every
intermediate quantity and every result, with the run manifest. Used by the
weekly workflow (derived files and the update log), by the paper's
experiments, and for archiving. --as-of reconstructs the run as the
dashboard stood on a past date (weekly replay).

Usage:
    python scripts/export_run.py [--as-of 2025-01-18] [--out run.xlsx]
                                 [--derived DIR] [--jurisdiction National]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SSDM_EDITOR", "1")

import logging  # noqa: E402
logging.getLogger("streamlit").setLevel(logging.ERROR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def collect(jurisdiction: str = "National") -> dict:
    from modules import state, feeds, live_subcriteria as ls
    from modules import scenario_engine as se, ssdm_core as sc
    from modules.agent_engine import AGENTS, CORE4, run_agent

    s = state.settings()
    lam, a, b, g = s["lambda"], s["alpha"], s["beta"], s["gamma"]
    out = {}
    ref = ls.reference_week(jurisdiction)
    out["Manifest"] = pd.DataFrame([{"key": k, "value": json.dumps(v, default=str)} for k, v in {
        "reference_week": str(ref)[:10] if ref is not None else "",
        "jurisdiction": jurisdiction, "context_key": state.context_key(),
        "inputs_store": state.backend().name}.items()])
    out["Settings"] = pd.DataFrame([{"setting": k, "value": v} for k, v in s.items()])
    out["Feeds"] = feeds.data_status()
    out["Historical_Stats"] = pd.DataFrame(ls.historical_stats(jurisdiction, 1, s["history_window"])).T.reset_index(names="criterion")
    adm = se.admission_table()
    adm["f_s 95% interval"] = adm["f_s 95% interval"].astype(str)
    out["Admission"] = adm
    out["Emphasis"] = pd.concat([se.emphasis_multipliers(x) for x in se.scenario_ids(True)], ignore_index=True)
    out["Limit_Overrides"] = pd.DataFrame([
        {"Scenario": x, "Agent": k, "Criterion": c, "Limit": v}
        for x in se.scenario_ids(True) for k in AGENTS
        for c, v in se.scenario_limit_overrides(x, k, jurisdiction).items()])
    out["Probabilities"] = se.probability_table(lam, a, b, g)
    prof_rows, strat_rows, it_rows = [], [], []
    for k in AGENTS:
        for scen in [None] + se.scenario_ids(True):
            prof, res, w, wprov, lim, lprov = run_agent(k, jurisdiction, CORE4, scen)
            env = scen or "Baseline"
            if scen is None:
                for _, r in prof.iterrows():
                    for c in AGENTS[k]["criteria"]:
                        if c in prof.columns:
                            prof_rows.append({"Agent": k, "Disease": r["Disease Type"], "Criterion": c,
                                              "Value": pd.to_numeric(r[c], errors="coerce")})
            rk = res.ranks()
            for d in res.order:
                rec = res.iterations[res.tiers[d] - 1]
                strat_rows.append({"Agent": k, "Environment": env, "Disease": d, "Tier": res.tiers[d],
                                   "Rank": rk[d], "S_i": res.scores[d], "K_i at the iteration forming its tier": rec.k_weighted[d],
                                   "Target": rec.target == d, "Weights": wprov, "Limits": lprov})
            for d in res.unstratified:
                strat_rows.append({"Agent": k, "Environment": env, "Disease": d, "Tier": None,
                                   "Rank": None, "S_i": None, "K_i at the iteration forming its tier": None, "Target": False,
                                   "Weights": wprov, "Limits": lprov})
            for rec in res.iterations:
                for c in rec.limits:
                    it_rows.append({"Agent": k, "Environment": env, "Iteration": rec.iteration,
                                    "Criterion": c, "Limit": rec.limits[c], "Source": rec.limit_source[c],
                                    "Position": rec.limit_positions.get(c), "Theta": rec.thetas.get(c),
                                    "Delta": rec.delta, "Margin": res.separations.get(rec.iteration)})
    out["Profiles"] = pd.DataFrame(prof_rows)
    out["Stratifications"] = pd.DataFrame(strat_rows)
    out["Iterations"] = pd.DataFrame(it_rows)
    out["Stability"] = pd.concat([se.stability_table(k, jurisdiction, lam, a, b, g).assign(Agent=k)
                                  for k in AGENTS], ignore_index=True)
    out["Typology"] = pd.concat([sc.quadrants(sc.indicators(k, jurisdiction, lam, a, b, g),
                                              s["theta_si"]).assign(Agent=k) for k in AGENTS],
                                ignore_index=True)
    for tag, base in (("Expected", False), ("Baseline", True)):
        M, res, wprov = sc.cross_agent_run(jurisdiction, lam, a, b, g, "", base)
        out[f"Second_Order_{tag}"] = M.reset_index(names="Disease")
        t = res.table()
        t["Margin to next tier"] = t["Tier"].map(res.separations)
        t["W_g provenance"] = wprov
        out[f"Final_{tag}"] = t
    return out


def update_log(prev: dict, cur: dict) -> pd.DataFrame:
    """What changed since the previous weekly run: episode counts, scenario
    weights, historical limits and final tiers."""
    rows = []

    def cmp(sheet, key, cols):
        if sheet not in prev or sheet not in cur:
            return
        a = prev[sheet].set_index(key) if key in prev[sheet] else None
        b = cur[sheet].set_index(key)
        if a is None:
            return
        for i in b.index:
            for c in cols:
                if c in b.columns and c in a.columns and i in a.index:
                    va, vb = a.at[i, c], b.at[i, c]
                    try:
                        same = np.isclose(float(va), float(vb), rtol=1e-9, atol=1e-12)
                    except Exception:
                        same = str(va) == str(vb)
                    if not same:
                        rows.append({"Quantity": f"{sheet}: {c}", "Item": i, "Previous": va, "Current": vb})
    cmp("Admission", "Scenario ID", ["n_s", "Y_s", "f_s"])
    cmp("Probabilities", "Scenario ID", ["p_s"])
    cmp("Historical_Stats", "criterion", ["p50", "p90"])
    cmp("Final_Expected", "Alternative", ["Tier", "Overall Rank"])
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--derived", default=None)
    ap.add_argument("--jurisdiction", default="National")
    args = ap.parse_args()
    from modules import live_connectors as lc
    if args.as_of:
        lc.AS_OF = pd.Timestamp(args.as_of)
    res = collect(args.jurisdiction)
    if args.derived:
        d = Path(args.derived)
        d.mkdir(parents=True, exist_ok=True)
        prev = {}
        for f in d.glob("*.csv"):
            if f.stem != "update_log":
                prev[f.stem] = pd.read_csv(f)
        for name, df in res.items():
            df.to_csv(d / f"{name}.csv", index=False)
        log = update_log(prev, {k: pd.read_csv(d / f"{k}.csv") for k in res})
        log.to_csv(d / "update_log.csv", index=False)
        print(f"derived files written to {d}; {len(log)} change(s) since the previous run")
    if args.out:
        with pd.ExcelWriter(args.out, engine="openpyxl") as w:
            for name, df in res.items():
                df.to_excel(w, sheet_name=name[:31], index=False)
        print(f"run exported to {args.out}")


if __name__ == "__main__":
    main()
