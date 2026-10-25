import numpy as np
import pandas as pd
from modules.utils_numeric import parse_numeric

AGENT_TITLE = "Agent 6: Equity & Vulnerable Populations"

SUBCRITERIA = [
    {"id": "equity_vulnerable_population_impact_|_pediatric_hospitalization_rate_per_100k_for_ages_5", "name": "Pediatric Hosp Rate (<5)", "kind": "cost"},
    {"id": "equity_vulnerable_population_impact_|_geriatric_hospitalization_rate_per_100k_for_ages_65", "name": "Geriatric Hosp Rate (65+)", "kind": "cost"},
    {"id": "equity_vulnerable_population_impact_|_highest_racial_disparity_rate_ratio", "name": "Racial Disparity Ratio", "kind": "cost"}
]

def build_agent6_profile(wide_df: pd.DataFrame) -> pd.DataFrame:
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
