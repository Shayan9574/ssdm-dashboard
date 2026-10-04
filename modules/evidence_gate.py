"""
evidence_gate.py

The automated Evidence Gate: closes the external evidence loop inside
the dashboard for end users with no access to code or files. Every
retrieved evidence row passes four checks; rows passing ALL of them
enter the Evidence Ledger (a sheet written directly to the workbook on
Drive, persisting across sessions) and apply as a clearly labeled
overlay on the affected subcriterion values; the stratification then
recomputes in session. Rows failing any check are quarantined with the
reasons stated. The curated baseline is never overwritten; the overlay
is one toggle away from comparison at all times.

Gate 1, source: the source must resolve to a trusted publisher
        (federal .gov, WHO, major journals); grounding redirect links
        are resolved to their final domain, with a stated weaker
        fallback on the source title when resolution is impossible.
Gate 2, schema and plausibility: the metric must map to a known
        subcriterion, the value must parse, the scale is harmonized to
        the stored column's convention (factor 1, 0.01, or 100), and
        the harmonized value must fall within an order of magnitude of
        the column's observed range (percent metrics within 0 to 100).
Gate 3, definition: deterministic screen against denominator traps
        (for example in hospital mortality presented as case fatality
        ratio), plus a model audit against the Data Dictionary
        definition when a key is available.
Gate 4, corroboration: an independent grounded query must return the
        same quantity within tolerance (default 25 percent relative).
"""

import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

from pathlib import Path
from modules.load_data import get_data_filepath
from modules.utils_numeric import parse_numeric
from modules import genai

LEDGER_SHEET = "Evidence Ledger"          # legacy name, no longer written
def _ledger_path():
    """The ledger lives in its own small file beside the master workbook,
    so the live app never rewrites the large master (Drive safe, atomic)."""
    # resolve() follows the Colab symlink so the ledger lands on Drive, beside
    # the real master, and survives runtime resets.
    return Path(get_data_filepath()).resolve().parent / "evidence_ledger.csv"
TOLERANCE = 0.25
TRUSTED_PATTERNS = [
    r"\.gov(/|$)", r"who\.int", r"ncbi\.nlm\.nih\.gov", r"pubmed",
    r"jamanetwork\.com", r"nejm\.org", r"thelancet\.com",
    r"academic\.oup\.com", r"cambridge\.org", r"nature\.com",
    r"sciencedirect\.com", r"cidrap\.umn\.edu",
]
TITLE_PATTERNS = [r"\bCDC\b", r"\bMMWR\b", r"\bWHO\b", r"HAN", r"MMWR"]
DENOMINATOR_TRAPS = {
    "fatality": ["hospitalized", "in-hospital", "in hospital",
                 "hospitalizations resulting", "of hospitalized",
                 "known outcomes"],
    "incidence": ["among hospitalized"],
}


# ----------------------------------------------------------------------
# gate 1: source
# ----------------------------------------------------------------------

def check_source(url: str, title: str) -> Tuple[bool, str]:
    final = url
    try:
        r = requests.head(url, allow_redirects=True, timeout=10)
        final = r.url
    except Exception:
        try:
            r = requests.get(url, allow_redirects=True, timeout=10, stream=True)
            final = r.url
        except Exception:
            final = None
    if final:
        if any(re.search(p, final, re.I) for p in TRUSTED_PATTERNS):
            return True, f"resolved to trusted domain: {final.split('/')[2]}"
        return False, f"resolved domain not on the trusted list: {final.split('/')[2] if '://' in final else final}"
    if any(re.search(p, str(title)) for p in TITLE_PATTERNS):
        return True, "URL unresolvable; trusted publisher matched on title (weaker)"
    return False, "URL unresolvable and title matches no trusted publisher"


# ----------------------------------------------------------------------
# gate 2: schema, scale harmonization, plausibility
# ----------------------------------------------------------------------

def _match_criterion(maps_to: str, criteria: List[str]) -> Optional[str]:
    m = str(maps_to).strip().lower()
    if m.startswith("new"):
        return None
    for c in criteria:
        cl = c.lower()
        if m == cl or m in cl or cl in m:
            return c
    return None


def check_schema(row: dict, profile: pd.DataFrame,
                 criteria: List[str]) -> Tuple[bool, str, Optional[str],
                                               Optional[float]]:
    crit = _match_criterion(row.get("maps_to", ""), criteria)
    if crit is None:
        return False, "maps to no existing subcriterion (informational row)", None, None
    v = parse_numeric(row.get("value"))
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False, "value does not parse to a number", crit, None
    col = pd.to_numeric(profile[crit], errors="coerce").dropna()
    if col.empty:
        return True, "no stored values to bound against; accepted on parse", crit, float(v)
    med = float(col.median())
    best_f, best_d = 1.0, float("inf")
    for f in (1.0, 0.01, 100.0):
        cand = v * f
        if cand <= 0 or med <= 0:
            d = abs(cand - med)
        else:
            d = abs(math.log10(cand) - math.log10(med))
        if d < best_d:
            best_d, best_f = d, f
    hv = v * best_f
    lo, hi = float(col.min()) / 10.0, float(col.max()) * 10.0
    if "%" in crit and not (0.0 <= hv <= 100.0) and not (0.0 <= hv <= 1.0):
        return False, f"harmonized value {hv:g} outside percent bounds", crit, None
    if not (lo <= hv <= hi):
        return False, (f"harmonized value {hv:g} outside plausibility bounds "
                       f"[{lo:g}, {hi:g}]"), crit, None
    note = "" if best_f == 1.0 else f" (scale harmonized, factor {best_f:g})"
    return True, f"parsed {v:g}, applied as {hv:g}{note}", crit, float(hv)


# ----------------------------------------------------------------------
# gate 3: definition
# ----------------------------------------------------------------------

def check_definition(row: dict, criterion: str,
                     model: str = "gemini-2.5-flash") -> Tuple[bool, str]:
    ctx = " ".join(str(row.get(k, "")) for k in
                   ("metric", "use", "timeframe", "source_title")).lower()
    for family, traps in DENOMINATOR_TRAPS.items():
        if family in criterion.lower():
            for t in traps:
                if t in ctx:
                    return False, (f"denominator trap: context mentions "
                                   f"'{t}', which contradicts the "
                                   f"population level definition of "
                                   f"{criterion}")
    if genai.gemini_available():
        try:
            prompt = (
                "Answer ONLY with JSON {\"match\": true|false, \"reason\": "
                "\"<one sentence>\"}. Does this evidence item's definition "
                f"match the subcriterion '{criterion}' understood as a "
                "population level rate for the named disease (not restricted "
                "to hospitalized patients unless the subcriterion says so)?\n"
                f"Item: {json.dumps({k: str(row.get(k, '')) for k in ('disease', 'metric', 'value', 'unit', 'timeframe', 'use')})}")
            raw = genai._with_fallback(genai.call_gemini,
                                       os.environ.get("GEMINI_API_KEY", ""),
                                       model, prompt, temperature=0.0)
            txt = re.sub(r"```(json)?", "", raw).strip()
            data = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
            if not bool(data.get("match", False)):
                return False, "model audit: " + str(data.get("reason", ""))
            return True, "deterministic screen passed; model audit agreed"
        except Exception as e:
            return True, f"deterministic screen passed; model audit skipped ({e})"
    return True, "deterministic screen passed; model audit unavailable"


# ----------------------------------------------------------------------
# gate 4: corroboration
# ----------------------------------------------------------------------

def check_corroboration(row: dict, harmonized: float,
                        profile_scale_hint: str,
                        model: str = "gemini-2.5-flash") -> Tuple[bool, str]:
    if not genai.gemini_available():
        return False, "no key available for the independent corroboration query"
    try:
        prompt = (
            "Using live web search of official sources only (CDC, WHO, "
            "peer reviewed), report the value of this quantity. Answer ONLY "
            "with JSON {\"value\": <number>, \"unit\": \"<unit>\", "
            "\"source_title\": \"<title>\"}.\n"
            f"Quantity: {row.get('metric')} for {row.get('disease')}, "
            f"{row.get('timeframe')}, in {row.get('unit')}.")
        text, _ = genai._with_fallback(genai.call_gemini_grounded,
            os.environ.get("GEMINI_API_KEY", ""), model, prompt, 0.0)
        txt = re.sub(r"```(json)?", "", text).strip()
        data = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
        v2 = parse_numeric(data.get("value"))
        if v2 is None:
            return False, "corroboration query returned no numeric value"
        v1 = parse_numeric(row.get("value"))
        rel = abs(v2 - v1) / max(abs(v1), 1e-9)
        if rel <= TOLERANCE:
            return True, (f"independent query returned {v2:g} "
                          f"({data.get('source_title', '')}); relative "
                          f"difference {rel:.1%} within tolerance")
        return False, (f"independent query returned {v2:g}; relative "
                       f"difference {rel:.1%} exceeds {TOLERANCE:.0%}")
    except Exception as e:
        return False, f"corroboration failed to complete ({e})"


# ----------------------------------------------------------------------
# the gate, the ledger, and the overlay
# ----------------------------------------------------------------------

def run_gate(rows: List[dict], profile: pd.DataFrame, criteria: List[str],
             agent_key: str, model: str = "gemini-2.5-flash") -> pd.DataFrame:
    out = []
    for row in rows:
        g1, r1 = check_source(row.get("source_url", ""), row.get("source_title", ""))
        g2, r2, crit, hv = check_schema(row, profile, criteria)
        g3, r3 = (check_definition(row, crit, model) if crit
                  else (False, "no criterion to audit"))
        g4, r4 = (check_corroboration(row, hv, "", model)
                  if (g1 and g2 and g3) else
                  (False, "skipped: an earlier gate failed"))
        status = "validated" if (g1 and g2 and g3 and g4) else (
            "informational" if crit is None else "quarantined")
        out.append({
            "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "agent": agent_key, "disease": row.get("disease"),
            "subcriterion": crit or row.get("maps_to"),
            "raw_value": row.get("value"), "applied_value": hv,
            "unit": row.get("unit"), "timeframe": row.get("timeframe"),
            "source_title": row.get("source_title"),
            "gate1_source": f"{'PASS' if g1 else 'FAIL'}: {r1}",
            "gate2_schema": f"{'PASS' if g2 else 'FAIL'}: {r2}",
            "gate3_definition": f"{'PASS' if g3 else 'FAIL'}: {r3}",
            "gate4_corroboration": f"{'PASS' if g4 else 'FAIL'}: {r4}",
            "status": status,
        })
    return pd.DataFrame(out)


def append_ledger(report: pd.DataFrame) -> int:
    """Persists the gate report to evidence_ledger.csv beside the master
    workbook (on Drive in Colab): read, concatenate, write to a temporary
    file, atomically replace. The master workbook is never touched."""
    import os as _os
    lp = _ledger_path()
    try:
        existing = pd.read_csv(lp)
    except Exception:
        existing = pd.DataFrame()
    combined = pd.concat([existing, report], ignore_index=True)
    tmp = lp.with_suffix(".tmp.csv")
    combined.to_csv(tmp, index=False)
    _os.replace(tmp, lp)
    load_ledger.clear()
    return len(report)


@st.cache_data(ttl=21600)
def load_ledger() -> pd.DataFrame:
    try:
        return pd.read_csv(_ledger_path())
    except Exception:
        return pd.DataFrame()


def overlay_for(agent_key: str, criteria: List[str]) -> Dict[Tuple[str, str], float]:
    """(disease, criterion) -> validated applied value; newest wins."""
    led = load_ledger()
    if led.empty:
        return {}
    led = led[(led["agent"] == agent_key) & (led["status"] == "validated")
              & led["subcriterion"].isin(criteria)]
    out = {}
    for _, r in led.iterrows():
        v = pd.to_numeric(r["applied_value"], errors="coerce")
        if pd.notna(v):
            out[(str(r["disease"]), str(r["subcriterion"]))] = float(v)
    return out


def apply_overlay(profile: pd.DataFrame, agent_key: str,
                  criteria: List[str]) -> Tuple[pd.DataFrame, int]:
    ov = overlay_for(agent_key, criteria)
    if not ov:
        return profile, 0
    prof = profile.copy()
    n = 0
    for (dis, crit), v in ov.items():
        m = prof["Disease Type"].astype(str) == dis
        if m.any() and crit in prof.columns:
            prof.loc[m, crit] = v
            n += 1
    return prof, n
