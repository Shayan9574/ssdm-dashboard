import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from modules.utils_numeric import parse_numeric

AGENT_TITLE = "Agent 1: Epidemiological Burden & Severity"

SUBCRITERIA = [
    {"id": "epidemiological_burden_|_annual_incidence_per_100k", "name": "Annual Incidence Rate (per 100k)", "kind": "cost"},
    {"id": "epidemiological_burden_|_annual_hospitalization_rate_per_100k", "name": "Annual Hospitalization Rate (per 100k)", "kind": "cost"},
    {"id": "epidemiological_burden_|_icu_rate_of_hospitalized", "name": "ICU Rate (% of Hosp)", "kind": "cost"},
    {"id": "epidemiological_burden_|_case_fatality_ratio", "name": "Case-Fatality Ratio (%)", "kind": "cost"}
]

def build_agent1_profile(wide_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["Disease Type"] = wide_df["Disease Type"].astype(str)
    
    for sc in SUBCRITERIA:
        col_id = sc["id"]
        name = sc["name"]
        if col_id in wide_df.columns:
            df[name] = wide_df[col_id].apply(parse_numeric)
        else:
            df[name] = np.nan
            
    return df

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """In cost space, a dominates b if a is >= b everywhere and > b somewhere (strictly worse/higher burden)."""
    return np.all(a >= b) and np.any(a > b)

def mosdm_tiers(df: pd.DataFrame, criteria_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    data = df[criteria_cols].values
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
                    explanations.append(f"**{df.iloc[j]['Disease Type']}** dominates **{df.iloc[i]['Disease Type']}** across burden criteria.")
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