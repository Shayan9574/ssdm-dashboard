import numpy as np
import pandas as pd
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
