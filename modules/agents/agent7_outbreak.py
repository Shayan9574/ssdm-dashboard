import numpy as np
import pandas as pd

from modules.live_connectors import get_rt_surveillance

AGENT_TITLE = "Agent 7: Outbreak Dynamics & Surveillance"

SPREAD_MAP = {"widespread": 3.0, "regional": 2.0, "local": 1.0, "sporadic": 0.5}


def rt_current(jurisdiction: str = "National", lag_complete_weeks: int = 1):
    """(disease key -> {median, p_growing, as_of_date} for the current values,
    list of freshness notes for the values treated as missing because older
    than the freshness limit relative to the dashboard's reference week)."""
    from modules import live_subcriteria as ls
    rt_data = get_rt_surveillance(jurisdiction=jurisdiction, lag_complete_weeks=lag_complete_weeks)
    ref = ls.reference_week(jurisdiction, lag_complete_weeks)
    cur, notes = {}, []
    for key, rec in rt_data.items():
        age = ((ref - pd.Timestamp(rec["as_of_date"])).days / 7.0) if ref is not None else 0.0
        if age > ls.MAX_AGE_WEEKS:
            notes.append({"Disease": key, "Criterion": "Rt and growth probability",
                          "Latest date": rec["as_of_date"], "Age (weeks)": round(age, 1),
                          "Status": "stale, treated as missing"})
        else:
            cur[key] = rec
    return cur, notes


def _spread_score(x) -> float:
    """Ordinal geographic spread score from the curated status (widespread 3,
    regional 2, localized 1, sporadic 0.5); an unrecognized or unknown status
    stays missing."""
    x = str(x).lower()
    for key in ("widespread", "regional", "local", "sporadic"):
        if key in x:
            return SPREAD_MAP[key]
    return np.nan


def build_agent7_profile(wide_df: pd.DataFrame, jurisdiction: str = "National",
                         lag_complete_weeks: int = 1) -> pd.DataFrame:
    df = pd.DataFrame()
    df["Disease Type"] = wide_df["Disease Type"].astype(str)
    spread_col = "outbreak_dynamics_risk_|_geographic_spread_status"
    df["Geographic Spread Status"] = (wide_df[spread_col].astype(str)
                                      if spread_col in wide_df.columns else "")
    df["Geographic Spread Risk (Score)"] = df["Geographic Spread Status"].apply(_spread_score)
    rt_data, _ = rt_current(jurisdiction, lag_complete_weeks)
    df["Effective Reproduction Number (Rt)"] = np.nan
    df["Probability of Outbreak Growth"] = np.nan
    for idx, row in df.iterrows():
        d = str(row["Disease Type"]).strip().lower()
        key = next((k for k in rt_data if k.lower() in d or d in k.lower()), None)
        if key is None:
            continue
        df.at[idx, "Effective Reproduction Number (Rt)"] = rt_data[key].get("median", np.nan)
        df.at[idx, "Probability of Outbreak Growth"] = rt_data[key].get("p_growing", np.nan)
    return df
