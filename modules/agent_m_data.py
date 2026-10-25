"""
agent_m_data.py

The active decision matrix: the curated baseline (Baseline Data sheet) with
the live subcriteria of the target week appended for display. The agents
build their profiles from the curated columns and attach their own live
columns (modules.live_subcriteria), so the displayed matrix and the engine
read the same values.
"""
import numpy as np
import pandas as pd

from modules import state
from modules.load_data import load_wide
from modules.utils_numeric import parse_numeric, parse_binary, is_missing


@state.cached
def build_harmonized_long_table() -> pd.DataFrame:
    df_wide = load_wide()
    id_cols = ["Disease Type", "season/year"]
    meta_cols = [c for c in df_wide.columns if c.startswith("pathogen_metadata_")]
    value_cols = [c for c in df_wide.columns if c not in id_cols and c not in meta_cols]
    df_long = df_wide.melt(id_vars=id_cols + meta_cols, value_vars=value_cols,
                           var_name="criterion_id", value_name="raw_value")
    df_long["agent_domain"] = df_long["criterion_id"].apply(
        lambda x: str(x.split("_|_")[0]) if "_|_" in str(x) else "general")
    df_long["subcriterion_label"] = df_long["criterion_id"].apply(
        lambda x: str(x.split("_|_")[1]) if "_|_" in str(x) else str(x))
    df_long["numeric_value"] = [
        np.nan if is_missing(v) else (parse_binary(v) if "available_y_n" in str(c) else parse_numeric(v))
        for v, c in zip(df_long["raw_value"], df_long["criterion_id"])]
    df_long["raw_value"] = df_long["raw_value"].astype(str)
    return df_long


@state.cached
def build_hybrid_decision_matrix(jurisdiction: str = "National",
                                 lag_complete_weeks: int = 1) -> pd.DataFrame:
    from modules import live_subcriteria as ls
    df = load_wide().copy()
    panel = ls.live_panel(jurisdiction, lag_complete_weeks)
    for c in panel.columns:
        df[f"live | {c}"] = df["Disease Type"].map(panel[c].to_dict())
    df["surveillance_jurisdiction"] = jurisdiction
    ref = ls.reference_week(jurisdiction, lag_complete_weeks)
    df["live_reference_week"] = str(ref)[:10] if ref is not None else ""
    return df


def get_active_decision_matrix(jurisdiction: str = "National", lag_complete_weeks: int = 1) -> pd.DataFrame:
    return build_hybrid_decision_matrix(jurisdiction=jurisdiction, lag_complete_weeks=lag_complete_weeks)
