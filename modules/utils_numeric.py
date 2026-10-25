"""
utils_numeric.py

Parsing of curated evidence cells into fuzzy numbers and their crisp
representation (thesis Chapter 2, Eq. ch2_gmir).

  "13.8%", "$16B", "2.24x", "15,000"  -> crisp value v, trapezoid (v, v, v, v)
  "41.0% - 78.0%", "3 to 4"           -> interval [a, b], trapezoid (a, a, b, b)
  "<0.2%", "<1"                       -> interval [0, x] for a nonnegative
                                          quantity, trapezoid (0, 0, x, x)
  "~5"                                -> crisp 5

The crisp value entering the stratification is the graded mean
integration representation (GMIR) of the trapezoid, which for an interval
is its midpoint. The trapezoid itself is available for the fuzzy within
tier distances.
"""
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd

MISSING = {"nan", "none", "n/a", "null", "", "unknown", "tbd", "-", "nr", "np",
           "nc", "nn", ".", "not available", "na"}
_NUM = r"[-+]?\d*\.?\d+"
_UNIT = r"\s*(?:%|[bBmMkK]|x)?"
_RANGE = re.compile(rf"\$?({_NUM}){_UNIT}\s*(?:-|–|—|to)\s*\$?({_NUM}){_UNIT}")
_LESS = re.compile(rf"^\s*(?:<|≤|<=|less than)\s*\$?({_NUM})")
_SINGLE = re.compile(_NUM)


def is_missing(val) -> bool:
    if val is None:
        return True
    try:
        if pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass
    return str(val).strip().lower() in MISSING


def gmir(t: Tuple[float, float, float, float]) -> float:
    return (t[0] + 2.0 * t[1] + 2.0 * t[2] + t[3]) / 6.0


def parse_fuzzy(val) -> Optional[Tuple[float, float, float, float]]:
    if is_missing(val):
        return None
    if isinstance(val, (int, float, np.integer, np.floating)):
        v = float(val)
        return (v, v, v, v)
    s = str(val).strip().replace(",", "")
    m = _LESS.match(s)
    if m:
        x = float(m.group(1))
        return (0.0, 0.0, x, x) if x > 0 else (x, x, x, x)
    m = _RANGE.search(s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        lo, hi = min(a, b), max(a, b)
        return (lo, lo, hi, hi)
    m = _SINGLE.search(s.replace("~", ""))
    if m:
        v = float(m.group(0))
        return (v, v, v, v)
    return None


def parse_numeric(val) -> float:
    """Crisp value of a cell: the GMIR of its fuzzy representation."""
    t = parse_fuzzy(val)
    if t is None:
        return np.nan
    return float(t[0]) if t[0] == t[3] else float(gmir(t))


def parse_binary(val) -> float:
    """Yes / No indicators as 1 / 0."""
    if is_missing(val):
        return np.nan
    s = str(val).strip().lower()
    if s in {"yes", "y", "true", "1", "available"}:
        return 1.0
    if s in {"no", "n", "false", "0", "unavailable"}:
        return 0.0
    return parse_numeric(val)


def parse_binary_fuzzy(val) -> Optional[Tuple[float, float, float, float]]:
    v = parse_binary(val)
    return None if pd.isna(v) else (v, v, v, v)
