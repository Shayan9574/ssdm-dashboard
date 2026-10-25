import numpy as np
import pandas as pd
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
