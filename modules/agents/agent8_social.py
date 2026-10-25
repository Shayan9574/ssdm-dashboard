import numpy as np
import pandas as pd
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
