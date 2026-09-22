import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from modules.utils_numeric import parse_numeric

AGENT_TITLE = "Agent 2: Transmission & Susceptibility"

SUBCRITERIA = [
    {"id": "transmission_population_susceptibility_|_basic_reproduction_number_r0", "name": "Basic Reproduction Number (R0)", "kind": "cost"},
    {"id": "transmission_population_susceptibility_|_test_positivity_rate", "name": "Test Positivity Rate (%)", "kind": "cost"},
    {"id": "transmission_population_susceptibility_|_natural_infection_seroprevalence", "name": "Natural Infection Seroprevalence (%)", "kind": "benefit"},
    {"id": "transmission_population_susceptibility_|_asymptomatic_transmission_rate", "name": "Asymptomatic Transmission Rate (%)", "kind": "cost"},
    {"id": "transmission_population_susceptibility_|_household_secondary_attack_rate", "name": "Household Secondary Attack Rate (%)", "kind": "cost"}
]

def build_agent2_profile(wide_df: pd.DataFrame) -> pd.DataFrame:
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
                    explanations.append(f"**{df.iloc[j]['Disease Type']}** dominates **{df.iloc[i]['Disease Type']}** across transmission metrics.")
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