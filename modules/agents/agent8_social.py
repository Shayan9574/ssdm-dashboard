import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from modules.utils_numeric import parse_numeric

AGENT_TITLE = "Agent 8: Social & Economic Impact"

def build_agent8_profile(wide_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["Disease Type"] = wide_df["Disease Type"].astype(str)

    svi_col = "social_economic_impact_|_high_svi_area_burden_multiplier"
    cost_col = "social_economic_impact_|_estimated_annual_direct_medical_cost_b"

    if svi_col in wide_df.columns:
        df["Primary SVI Vulnerability Driver"] = wide_df[svi_col].astype(str)
    else:
        df["Primary SVI Vulnerability Driver"] = "Unknown"

    if cost_col in wide_df.columns:
        df["Estimated Annual Direct Medical Cost ($B)"] = wide_df[cost_col].apply(parse_numeric)
    else:
        df["Estimated Annual Direct Medical Cost ($B)"] = np.nan

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
                    explanations.append(f"**{df.iloc[j]['Disease Type']}** dominates **{df.iloc[i]['Disease Type']}** in economic burden.")
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