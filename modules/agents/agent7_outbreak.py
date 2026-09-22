import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from modules.utils_numeric import parse_numeric
from modules.live_connectors import get_rt_surveillance

AGENT_TITLE = "Agent 7: Outbreak Dynamics & Surveillance"

def build_agent7_profile(
    wide_df: pd.DataFrame,
    jurisdiction: str = "National",
    lag_complete_weeks: int = 1
) -> pd.DataFrame:
    df = pd.DataFrame()
    df["Disease Type"] = wide_df["Disease Type"].astype(str)

    spread_col = "outbreak_dynamics_risk_|_geographic_spread_status"
    if spread_col in wide_df.columns:
        df["Geographic Spread Status"] = wide_df[spread_col].astype(str)
    else:
        df["Geographic Spread Status"] = "Unknown"

    spread_map = {"widespread": 3.0, "regional": 2.0, "local": 1.0, "sporadic": 0.5, "unknown": 1.0}
    df["Geographic Spread Risk (Score)"] = df["Geographic Spread Status"].str.lower().map(lambda x: spread_map.get(x, 1.0))

    rt_data = get_rt_surveillance(jurisdiction=jurisdiction, lag_complete_weeks=lag_complete_weeks)

    df["Effective Reproduction Number (Rt)"] = np.nan
    df["Probability of Outbreak Growth"] = np.nan

    for idx, row in df.iterrows():
        d = str(row["Disease Type"]).strip()
        matched_key = None
        for k in rt_data.keys():
            if k.lower() in d.lower() or d.lower() in k.lower():
                matched_key = k
                break
        if matched_key and matched_key in rt_data:
            df.at[idx, "Effective Reproduction Number (Rt)"] = rt_data[matched_key].get("median", np.nan)
            df.at[idx, "Probability of Outbreak Growth"] = rt_data[matched_key].get("p_growing", np.nan)

    return df

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return np.all(a >= b) and np.any(a > b)

def mosdm_tiers(df: pd.DataFrame, criteria_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    data = df[criteria_cols].fillna(0).values
    n = len(df)
    remaining = set(range(n))
    tier_map = {}
    explanations = []
    current_tier = 1

    while remaining:
        non_dominated = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i != j and dominates(data[j], data[i]):
                    dominated = True
                    explanations.append(f"**{df.iloc[j]['Disease Type']}** dominates **{df.iloc[i]['Disease Type']}** across outbreak velocity metrics.")
                    break
            if not dominated:
                non_dominated.append(i)

        if not non_dominated:
            for i in remaining:
                tier_map[i] = current_tier
            break

        for i in non_dominated:
            tier_map[i] = current_tier
            remaining.remove(i)
        current_tier += 1

    out_df = df.copy()
    out_df["Tier"] = [tier_map[i] for i in range(n)]
    return out_df[["Disease Type", "Tier"] + criteria_cols].sort_values("Tier"), explanations

def weighted_tiebreak(df: pd.DataFrame, weights: Dict[str, float], criteria_cols: List[str]) -> pd.DataFrame:
    out_df = df.copy()
    norm_matrix = pd.DataFrame(index=df.index)
    
    for c in criteria_cols:
        col_vals = pd.to_numeric(df[c], errors="coerce").fillna(0)
        c_min, c_max = col_vals.min(), col_vals.max()
        if c_max > c_min:
            norm_matrix[c] = (col_vals - c_min) / (c_max - c_min)
        else:
            norm_matrix[c] = 1.0 if c_max > 0 else 0.0

    scores = np.zeros(len(df))
    for c in criteria_cols:
        scores += norm_matrix[c].values * weights.get(c, 0.0)

    out_df["weighted_score"] = np.round(scores, 4)
    return out_df.sort_values("weighted_score", ascending=False)

def scenario_weights_emphasis(criteria_cols: List[str], emphasis_keywords: List[str]) -> Dict[str, float]:
    weights = {}
    for c in criteria_cols:
        c_clean = c.lower()
        if any(k in c_clean for k in emphasis_keywords):
            weights[c] = 3.0
        else:
            weights[c] = 1.0
            
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}