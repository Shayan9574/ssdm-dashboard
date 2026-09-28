from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

from modules.load_data import load_wide, load_data_dictionary
from modules.utils_numeric import parse_numeric, parse_binary, is_missing
from modules.live_connectors import (
    get_rt_surveillance,
    get_hospital_respiratory_surveillance,
    get_ed_visit_surveillance,
    get_wastewater_surveillance,
)

@st.cache_data(ttl=21600)
def build_harmonized_long_table() -> pd.DataFrame:
    """
    Constructs the harmonized long table under strict zero-imputation rules.
    Enforces Arrow-compatible scalar types on all columns.
    """
    df_wide = load_wide()
    dict_df = load_data_dictionary()

    id_cols = ["Disease Type", "season/year"]
    meta_cols = [c for c in df_wide.columns if c.startswith("pathogen_metadata_")]
    value_cols = [c for c in df_wide.columns if c not in id_cols and c not in meta_cols]

    df_long = df_wide.melt(
        id_vars=id_cols + meta_cols,
        value_vars=value_cols,
        var_name="criterion_id",
        value_name="raw_value"
    )

    df_long["agent_domain"] = df_long["criterion_id"].apply(
        lambda x: str(x.split("_|_")[0]) if "_|_" in str(x) else "general"
    )
    df_long["subcriterion_label"] = df_long["criterion_id"].apply(
        lambda x: str(x.split("_|_")[1]) if "_|_" in str(x) else str(x)
    )

    def _parse_cell(val, crit_id):
        if is_missing(val):
            return np.nan
        if "available_y_n" in str(crit_id):
            return parse_binary(val)
        return parse_numeric(val)

    df_long["numeric_value"] = [
        _parse_cell(v, c) for v, c in zip(df_long["raw_value"], df_long["criterion_id"])
    ]
    df_long["numeric_value"] = pd.to_numeric(df_long["numeric_value"], errors="coerce")
    df_long["raw_value"] = df_long["raw_value"].astype(str)
    df_long["Disease Type"] = df_long["Disease Type"].astype(str)
    df_long["season/year"] = df_long["season/year"].astype(str)

    return df_long

@st.cache_data(ttl=21600)
def build_hybrid_decision_matrix(
    jurisdiction: str = "National",
    lag_complete_weeks: int = 1
) -> pd.DataFrame:
    """
    Constructs the active decision matrix.
    Maintains historical annual baselines while appending active surveillance metrics
    as explicit columns for the selected jurisdiction.
    """
    df = load_wide()
    
    hosp = get_hospital_respiratory_surveillance(jurisdiction, lag_complete_weeks)
    ed = get_ed_visit_surveillance(jurisdiction, lag_complete_weeks)
    rt = get_rt_surveillance(jurisdiction, lag_complete_weeks)
    ww = get_wastewater_surveillance(jurisdiction, lag_complete_weeks)

    # Initialize live surveillance columns
    df["live_weekly_hosp_adm_per_100k"] = np.nan
    df["live_icu_share_pct"] = np.nan
    df["live_ed_visit_pct"] = np.nan
    df["live_rt_median"] = np.nan
    df["live_prob_growth"] = np.nan
    df["live_wastewater_conc"] = np.nan
    df["surveillance_jurisdiction"] = jurisdiction

    for idx, row in df.iterrows():
        d = str(row["Disease Type"]).strip()
        
        if d == "COVID-19":
            if hosp:
                df.at[idx, "live_weekly_hosp_adm_per_100k"] = hosp.get("covid_adm_per100k", np.nan)
                df.at[idx, "live_icu_share_pct"] = hosp.get("covid_icu_rate", np.nan)
            if ed:
                df.at[idx, "live_ed_visit_pct"] = ed.get("covid_ed_pct", np.nan)
            if "COVID-19" in rt:
                df.at[idx, "live_rt_median"] = rt["COVID-19"].get("median", np.nan)
                df.at[idx, "live_prob_growth"] = rt["COVID-19"].get("p_growing", np.nan)
            if ww:
                df.at[idx, "live_wastewater_conc"] = ww.get("viral_conc", np.nan)
                
        elif d == "Influenza":
            if hosp:
                df.at[idx, "live_weekly_hosp_adm_per_100k"] = hosp.get("flu_adm_per100k", np.nan)
                df.at[idx, "live_icu_share_pct"] = hosp.get("flu_icu_rate", np.nan)
            if ed:
                df.at[idx, "live_ed_visit_pct"] = ed.get("flu_ed_pct", np.nan)
            if "Influenza" in rt:
                df.at[idx, "live_rt_median"] = rt["Influenza"].get("median", np.nan)
                df.at[idx, "live_prob_growth"] = rt["Influenza"].get("p_growing", np.nan)
                
        elif d == "Respiratory Syncytial Virus (RSV)":
            if hosp:
                df.at[idx, "live_weekly_hosp_adm_per_100k"] = hosp.get("rsv_adm_per100k", np.nan)
                df.at[idx, "live_icu_share_pct"] = hosp.get("rsv_icu_rate", np.nan)
            if ed:
                df.at[idx, "live_ed_visit_pct"] = ed.get("rsv_ed_pct", np.nan)
            if "RSV" in rt:
                df.at[idx, "live_rt_median"] = rt["RSV"].get("median", np.nan)
                df.at[idx, "live_prob_growth"] = rt["RSV"].get("p_growing", np.nan)

    # PyArrow string casting on object columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)

    return df

def get_active_decision_matrix(jurisdiction: str = "National", lag_complete_weeks: int = 1) -> pd.DataFrame:
    return build_hybrid_decision_matrix(jurisdiction=jurisdiction, lag_complete_weeks=lag_complete_weeks)