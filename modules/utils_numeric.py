import re
import numpy as np
import pandas as pd

def is_missing(val) -> bool:
    """Checks whether a value is considered missing / null / unreportable."""
    if val is None:
        return True
    if pd.isna(val):
        return True
    s = str(val).strip().lower()
    return s in {"nan", "none", "n/a", "null", "", "unknown", "tbd", "-", "nr", "np", "nc", "nn", "."}

def parse_numeric(val) -> float:
    """
    Robustly parses floats from strings containing ranges ('41.0% - 78.0%', '$0.2B - $0.3B'),
    percentages ('13.8%'), dollar values ('$16B'), multipliers ('2.24x'),
    inequalities ('<0.2%'), or commas ('15,000').
    """
    if is_missing(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip().replace(",", "")
    if is_missing(s):
        return np.nan

    # 1. Match two-number ranges with optional units: $, %, B, M, x, etc.
    range_match = re.search(
        r"[\$]?([\d.]+)\s*(?:%|[bBmMkK]|x)?\s*[-–—~to/]+\s*[\$]?([\d.]+)\s*(?:%|[bBmMkK]|x)?",
        s
    )
    if range_match:
        try:
            low = float(range_match.group(1))
            high = float(range_match.group(2))
            return (low + high) / 2.0
        except ValueError:
            pass

    # 2. Match single numeric value (extracts float from '$16B', '<0.2%', '13.8%', '15000')
    single_match = re.search(r"[-+]?\d*\.?\d+", s)
    if single_match:
        try:
            return float(single_match.group(0))
        except ValueError:
            return np.nan

    return np.nan

def parse_binary(val) -> float:
    """Parses binary indicators into 1.0 (Yes/True) and 0.0 (No/False)."""
    if is_missing(val):
        return np.nan
    s = str(val).strip().lower()
    if s in {"yes", "y", "true", "1", "available"}:
        return 1.0
    if s in {"no", "n", "false", "0", "unavailable"}:
        return 0.0
    return parse_numeric(val)