import numpy as np
import pandas as pd
from modules.utils_numeric import parse_numeric

AGENT_TITLE = "Agent 3: Healthcare System Impact & Facility Strain"

SUBCRITERIA = [
    {"id": "healthcare_system_impact_|_avg_length_of_stay_days", "name": "Average Length of Stay (Days)", "kind": "cost"},
    {"id": "healthcare_system_impact_|_annual_ed_visit_rate_per_100k", "name": "Annual ED Visit Rate (per 100k)", "kind": "cost"}
]

def build_agent3_profile(wide_df: pd.DataFrame) -> pd.DataFrame:
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
