"""
evidence_gate.py

External evidence as an information layer for decision makers (paper
Section 5.4). Evidence retrieved by the AI assistant passes four checks and
is recorded in the evidence ledger with every check and reason. It NEVER
changes the decision matrix or any stratification: validated figures are
shown beside the dashboard's own values, with their source, date and unit,
and with a flag where they differ notably, so that policy makers see what
the wider literature reports and curators see where the evidence base may
need review. Values are never rescaled: a figure in a different unit or
definition is quarantined with the reason, not converted.

Gate 1, source: the source must resolve to a trusted publisher (federal
        .gov, WHO, major journals); a stated weaker fallback on the title
        applies when resolution is impossible.
Gate 2, schema and plausibility: the value must parse; when the metric
        maps to an existing subcriterion it must lie within an order of
        magnitude of the dashboard's values for that subcriterion (percent
        metrics within 0 to 100), otherwise the unit or definition differs.
Gate 3, definition: deterministic screen against denominator traps, plus a
        model audit against the subcriterion definition when a key exists.
Gate 4, corroboration: an independent grounded query must return the same
        quantity within tolerance (default 25 percent relative).
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from modules import state
from modules.utils_numeric import parse_numeric
from modules import genai

TOLERANCE = 0.25
DIFF_FLAG = 0.25

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
# gate 2: schema and plausibility (no rescaling)
# ----------------------------------------------------------------------

def _match_criterion(maps_to: str, criteria: List[str]) -> Optional[str]:
    m = str(maps_to).strip().lower()
    if not m or m.startswith("new"):
        return None
    for c in criteria:
        if m == c.lower():
            return c
    for c in criteria:
        cl = c.lower()
        if m in cl or cl in m:
            return c
    return None


def check_schema(row: dict, profile: pd.DataFrame, criteria: List[str]
                 ) -> Tuple[bool, str, Optional[str], Optional[float]]:
    crit = _match_criterion(row.get("maps_to", ""), criteria)
    v = parse_numeric(row.get("value"))
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False, "value does not parse to a number", crit, None
    if crit is None:
        return True, f"parsed {v:g} {row.get('unit', '')}; new indicator", None, float(v)
    if "%" in crit and not (0.0 <= v <= 100.0):
        return False, f"value {v:g} outside 0 to 100 for a percent subcriterion", crit, float(v)
    col = pd.to_numeric(profile[crit], errors="coerce").dropna() if crit in profile.columns else pd.Series(dtype=float)
    if len(col):
        lo, hi = float(col.min()), float(col.max())
        lo_b = lo / 10.0 if lo > 0 else lo - abs(hi - lo)
        hi_b = hi * 10.0 if hi > 0 else hi + abs(hi - lo)
        if not (lo_b <= v <= hi_b):
            return False, (f"value {v:g} more than an order of magnitude outside the "
                           f"dashboard values [{lo:g}, {hi:g}]: unit or definition differs"), crit, float(v)
    return True, f"parsed {v:g} {row.get('unit', '')}", crit, float(v)


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
# the gate and the ledger
# ----------------------------------------------------------------------

def run_gate(rows: List[dict], profile: pd.DataFrame, criteria: List[str],
             agent_key: str, model: str = "gemini-2.5-flash") -> pd.DataFrame:
    out = []
    prof = profile.set_index("Disease Type") if "Disease Type" in profile.columns else profile
    for row in rows:
        g1, r1 = check_source(row.get("source_url", ""), row.get("source_title", ""))
        g2, r2, crit, val = check_schema(row, profile, criteria)
        g3, r3 = (check_definition(row, crit, model) if crit
                  else (True, "new indicator: no subcriterion definition to audit"))
        g4, r4 = (check_corroboration(row, val, "", model) if (g1 and g2 and g3)
                  else (False, "skipped: an earlier check failed"))
        ok = g1 and g2 and g3 and g4
        status = ("validated" if crit else "validated new indicator") if ok else "quarantined"
        dash = np.nan
        dis = str(row.get("disease", ""))
        if crit and crit in prof.columns:
            hit = [d for d in prof.index if dis.lower() in str(d).lower() or str(d).lower() in dis.lower()]
            if hit:
                dash = pd.to_numeric(prof.at[hit[0], crit], errors="coerce")
        rel = abs(val - dash) / abs(dash) if (val is not None and dash == dash and dash not in (0, 0.0)) else np.nan
        out.append({
            "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "agent": agent_key, "disease": dis, "metric": row.get("metric"),
            "subcriterion": crit or "", "value": val, "raw_value": row.get("value"),
            "unit": row.get("unit"), "timeframe": row.get("timeframe"),
            "source_title": row.get("source_title"), "source_url": row.get("source_url"),
            "use": row.get("use"), "dashboard_value": dash if dash == dash else None,
            "relative_difference": rel if rel == rel else None,
            "differs_notably": bool(rel == rel and rel > DIFF_FLAG),
            "gate1_source": f"{'PASS' if g1 else 'FAIL'}: {r1}",
            "gate2_schema": f"{'PASS' if g2 else 'FAIL'}: {r2}",
            "gate3_definition": f"{'PASS' if g3 else 'FAIL'}: {r3}",
            "gate4_corroboration": f"{'PASS' if g4 else 'FAIL'}: {r4}",
            "status": status, "model": model,
        })
    return pd.DataFrame(out)


def append_ledger(report: pd.DataFrame) -> int:
    rows = json.loads(report.to_json(orient="records"))
    return state.append_ledger(rows)


def load_ledger() -> pd.DataFrame:
    return pd.DataFrame(state.ledger())


def external_evidence(agent_key: Optional[str] = None, include_new: bool = True) -> pd.DataFrame:
    """Validated external figures for decision makers, newest first; never
    used by the stratification."""
    led = load_ledger()
    if led.empty or "status" not in led.columns:
        return pd.DataFrame()
    ok = led["status"].isin(["validated", "validated new indicator"] if include_new else ["validated"])
    led = led[ok]
    if agent_key is not None and "agent" in led.columns:
        led = led[led["agent"] == agent_key]
    cols = [c for c in ["disease", "metric", "subcriterion", "value", "unit", "timeframe",
                        "dashboard_value", "relative_difference", "differs_notably",
                        "source_title", "source_url", "use", "retrieved_at", "agent"] if c in led.columns]
    return led.sort_values("retrieved_at", ascending=False)[cols].reset_index(drop=True)
