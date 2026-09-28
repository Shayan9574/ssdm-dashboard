from pathlib import Path
import pandas as pd
from modules.data_cache import read_excel_sheet
import streamlit as st

DATA_PATHS = [
    Path(__file__).resolve().parent.parent / "data" / "Research Data.xlsx",
    Path(__file__).resolve().parent.parent / "Research Data.xlsx",
    Path("data/Research Data.xlsx"),
    Path("Research Data.xlsx")
]

def get_data_filepath() -> Path:
    for p in DATA_PATHS:
        if p.exists():
            return p
    return DATA_PATHS[0]

@st.cache_data(ttl=21600)
def load_wide(file_path: str = None) -> pd.DataFrame:
    """
    Loads 'Baseline Data' from Research Data.xlsx, standardizes dual-tier headers,
    and enforces strict string typing on object columns for PyArrow compatibility.
    """
    fp = Path(file_path) if file_path else get_data_filepath()
    if not fp.exists():
        raise FileNotFoundError(f"Master file not found at: {fp}")

    raw = read_excel_sheet(fp, sheet_name="Baseline Data", header=None)

    domains = (
        raw.iloc[0]
        .ffill()
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    
    subcriteria = (
        raw.iloc[1]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    col_names = []
    for idx, (dom, sub) in enumerate(zip(domains, subcriteria)):
        if idx == 0:
            col_names.append("Disease Type")
        elif sub == "baseline_data_timeframe":
            col_names.append("season/year")
        elif dom in {"nan", ""} or sub in {"nan", ""} or "nan_|_nan" in f"{dom}_|_{sub}":
            col_names.append(f"_drop_{idx}")
        else:
            if sub in {"peak_seasonality_months", "geographic_spread_status"}:
                dom = "outbreak_dynamics_risk"
            elif sub in {"high_svi_area_burden_multiplier", "estimated_annual_direct_medical_cost_b"}:
                dom = "social_economic_impact"
            
            col_names.append(f"{dom}_|_{sub}")

    df = raw.iloc[2:].copy()
    df.columns = col_names

    df = df.dropna(subset=["Disease Type"]).copy()
    df["Disease Type"] = df["Disease Type"].astype(str).str.strip()
    df["season/year"] = df["season/year"].astype(str).str.strip()

    valid_cols = [c for c in df.columns if not c.startswith("_drop_")]
    df_clean = df[valid_cols].reset_index(drop=True)

    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            df_clean[col] = df_clean[col].astype(str)

    return df_clean

@st.cache_data(ttl=21600)
def load_data_dictionary(file_path: str = None) -> pd.DataFrame:
    """Loads criteria metadata definitions from Research Data.xlsx."""
    fp = Path(file_path) if file_path else get_data_filepath()
    if not fp.exists():
        raise FileNotFoundError(f"Master file not found at: {fp}")

    raw_dd = read_excel_sheet(fp, sheet_name="Data Dictionary")
    regime_df = raw_dd.iloc[:, [5, 6, 7]].dropna(subset=[raw_dd.columns[5]]).copy()
    regime_df.columns = ["criterion_label", "data_regime", "live_source"]
    regime_df = regime_df[~regime_df["criterion_label"].str.match(r"^\d+\.", na=False)].reset_index(drop=True)
    return regime_df

@st.cache_data(ttl=21600)
def load_citations(target_diseases: list = None, file_path: str = None) -> pd.DataFrame:
    """
    Loads research citations from the Citations sheet.
    Optionally filters by the selected disease list.
    """
    fp = Path(file_path) if file_path else get_data_filepath()
    if not fp.exists():
        raise FileNotFoundError(f"Master file not found at: {fp}")

    cits = read_excel_sheet(fp, sheet_name="Citations")
    cits.columns = [str(c).strip() for c in cits.columns]

    if not target_diseases:
        return cits

    def _matches_disease(val):
        cd = str(val).lower().strip()
        for td in target_diseases:
            t = str(td).lower().strip()
            if "covid" in t and "covid" in cd: return True
            if ("flu" in t or "influenza" in t) and ("flu" in cd or "influenza" in cd): return True
            if ("rsv" in t or "syncytial" in t) and ("rsv" in cd or "syncytial" in cd): return True
            if "mening" in t and "mening" in cd: return True
            if "tuber" in t and "tuber" in cd: return True
            if "salmon" in t and "salmon" in cd: return True
            if "lyme" in t and "lyme" in cd: return True
            if "hiv" in t and "hiv" in cd: return True
        return False

    filtered = cits[cits["Target Disease"].apply(_matches_disease)].reset_index(drop=True)
    return filtered

@st.cache_data(ttl=21600)
def load_derivations(target_diseases: list = None, file_path: str = None) -> pd.DataFrame:
    """Loads the Derivations sheet: the detailed calculation behind every
    derived baseline figure, for the dashboard's calculation audit view."""
    fp = Path(file_path) if file_path else get_data_filepath()
    if not fp.exists():
        raise FileNotFoundError(f"Master file not found at: {fp}")
    try:
        der = read_excel_sheet(fp, sheet_name="Derivations")
    except ValueError:
        return pd.DataFrame()
    der.columns = [str(c).strip() for c in der.columns]
    if target_diseases:
        tl = [str(t).lower() for t in target_diseases]
        der = der[der["Target Disease"].astype(str).str.lower().apply(
            lambda d: any(t in d or d in t for t in tl))].reset_index(drop=True)
    return der


@st.cache_data(ttl=21600)
def load_scenario_registry(file_path: str = None) -> pd.DataFrame:
    """Loads the Scenario Registry curated sheet (Section 10)."""
    fp = Path(file_path) if file_path else get_data_filepath()
    if not fp.exists():
        raise FileNotFoundError(f"Master file not found at: {fp}")
    reg = read_excel_sheet(fp, sheet_name="Scenario Registry")
    reg.columns = [str(c).strip() for c in reg.columns]
    return reg
