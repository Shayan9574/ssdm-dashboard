import numpy as np
import pandas as pd
from modules.utils_numeric import parse_numeric, parse_binary

AGENT_TITLE = "Agent 5: Prevention & Control Feasibility"

SUBCRITERIA = [
    {"id": "prevention_control_feasibility_|_vaccine_available_y_n", "name": "Vaccine Available", "kind": "benefit"},
    {"id": "prevention_control_feasibility_|_vaccine_effectiveness", "name": "Vaccine Effectiveness (%)", "kind": "benefit"},
    {"id": "prevention_control_feasibility_|_national_vaccine_coverage", "name": "National Vaccine Coverage (%)", "kind": "benefit"},
    {"id": "prevention_control_feasibility_|_npi_transmission_reduction", "name": "NPI Reduction (%)", "kind": "benefit"},
    {"id": "prevention_control_feasibility_|_target_demographic_vaccine_uptake", "name": "Target Demographic Uptake (%)", "kind": "benefit"}
]

def build_agent5_profile(wide_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["Disease Type"] = wide_df["Disease Type"].astype(str)
    
    for sc in SUBCRITERIA:
        col_id = sc["id"]
        name = sc["name"]
        if col_id in wide_df.columns:
            if "available_y_n" in col_id:
                df[name] = wide_df[col_id].apply(parse_binary)
            else:
                df[name] = wide_df[col_id].apply(parse_numeric)
        else:
            df[name] = np.nan
            
    return df
