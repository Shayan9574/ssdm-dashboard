import numpy as np
import pandas as pd
from modules.utils_numeric import parse_numeric, parse_binary

AGENT_TITLE = "Agent 4: Clinical Severity & Complexity"

SUBCRITERIA = [
    {"id": "clinical_severity_treatment_complexity_|_avg_days_from_onset_to_severe_disease", "name": "Avg Days from Onset to Severe", "kind": "benefit_inverted"},
    {"id": "clinical_severity_treatment_complexity_|_effective_treatment_window_days", "name": "Effective Treatment Window (Days)", "kind": "benefit_inverted"},
    {"id": "clinical_severity_treatment_complexity_|_standard_treatment_available_y_n", "name": "Standard Treatment Available", "kind": "benefit_inverted"},
    {"id": "clinical_severity_treatment_complexity_|_antimicrobial_resistance_prevalence", "name": "Antimicrobial Resistance Prev (%)", "kind": "cost"},
    {"id": "clinical_severity_treatment_complexity_|_severe_complication_rate", "name": "Severe Complication Rate (%)", "kind": "cost"}
]

def build_agent4_profile(wide_df: pd.DataFrame) -> pd.DataFrame:
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
