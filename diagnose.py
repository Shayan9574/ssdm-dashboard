import os
import sys
import datetime
from pathlib import Path
import pandas as pd
import numpy as np

print("=" * 80)
print("  SSDM DASHBOARD DIAGNOSTIC & SELF-TEST SUITE")
print("=" * 80)

# --- 1. EXCEL FILE DISCOVERY & PATH AUDIT ---
print("\n[1] CHECKING EXCEL FILE LOCATIONS ON DISK...")
possible_paths = [
    Path("data/Research Data.xlsx"),
    Path("Research Data.xlsx"),
    Path(__file__).resolve().parent / "data" / "Research Data.xlsx",
    Path(__file__).resolve().parent / "Research Data.xlsx"
]

found_files = []
for p in possible_paths:
    try:
        resolved = p.resolve()
        if p.exists() and resolved not in [f.resolve() for f in found_files]:
            found_files.append(p)
    except Exception:
        pass

if not found_files:
    print("  ❌ FATAL: No 'Research Data.xlsx' found in current directory or 'data/' folder!")
    sys.exit(1)

for f in found_files:
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f))
    size_mb = os.path.getsize(f) / (1024 * 1024)
    print(f"  📁 Found Workbook: {f.resolve()}")
    print(f"     -> Size: {size_mb:.2f} MB | Last Modified: {mtime}")

if len(found_files) > 1:
    print("\n  ⚠️ WARNING: Multiple copies of 'Research Data.xlsx' detected!")
    print(f"     Your app loads files in this order: {found_files[0].resolve()} will take priority.")
    print("     Ensure your latest edits are saved to that primary file.")

target_file = found_files[0]
print(f"\n  🎯 Active Master File: {target_file}")

# --- 2. RAW EXCEL BASELINE DATA INSPECTION ---
print("\n[2] INSPECTING RAW 'Baseline Data' SHEET...")
raw_excel = pd.read_excel(target_file, sheet_name="Baseline Data", header=None)
print(f"  Shape: {raw_excel.shape[0]} rows x {raw_excel.shape[1]} columns")

print("\n  Domain Row (Row 0, first 8):")
print("   ", raw_excel.iloc[0, :8].tolist())

print("\n  Subcriterion Row (Row 1, first 8):")
print("   ", raw_excel.iloc[1, :8].tolist())

print("\n  Raw Cell Values in Sheet for First 4 Pathogens:")
cols_header = raw_excel.iloc[1].tolist()
for r in range(2, min(len(raw_excel), 6)):
    d_name = raw_excel.iloc[r, 0]
    print(f"\n  --- Pathogen: {d_name} (Row {r}) ---")
    for c in range(min(len(cols_header), 12)):
        header = cols_header[c]
        val = raw_excel.iloc[r, c]
        is_empty = pd.isna(val) or str(val).strip() in {"", "nan", "None"}
        status = "❌ EMPTY / NaN" if is_empty else f"✅ {repr(val)}"
        print(f"    Col {c:02d} [{header}]: {status}")

# --- 3. NUMERIC UTILITY PARSER VERIFICATION ---
print("\n[3] TESTING NUMERIC PARSER (utils_numeric)...")
try:
    from modules.utils_numeric import parse_numeric, parse_binary
    test_cases = [
        "15000", "15,000", "1114.71 - 2108.82", "13.8%", "$16B", "<0.2%", "Yes", "No", "NR", None, np.nan
    ]
    for tc in test_cases:
        res = parse_numeric(tc)
        print(f"  parse_numeric({repr(tc):<25}) -> {res}")
except Exception as e:
    print(f"  ❌ Failed to import or run utils_numeric: {e}")

# --- 4. LOAD_WIDE DATA INGESTION ---
print("\n[4] TESTING load_wide() AND GENERATED COLUMN KEYS...")
try:
    from modules.load_data import load_wide
    df_wide = load_wide(str(target_file))
    print(f"  Successfully loaded wide table: {df_wide.shape[0]} rows x {df_wide.shape[1]} columns")
    print("\n  Generated Columns in df_wide:")
    for col in df_wide.columns:
        print(f"    • {col}")
except Exception as e:
    print(f"  ❌ load_wide() failed: {e}")
    sys.exit(1)

# --- 5. DOMAIN AGENTS TEST & COLUMN MATCH AUDIT ---
print("\n[5] AUDITING ALL 8 DOMAIN AGENTS...")

try:
    from modules.agents.agent1_epidemiology import build_agent1_profile, SUBCRITERIA as A1_CRIT
    from modules.agents.agent2_transmission import build_agent2_profile, SUBCRITERIA as A2_CRIT
    from modules.agents.agent3_healthcare_impact import build_agent3_profile, SUBCRITERIA as A3_CRIT
    from modules.agents.agent4_clinical import build_agent4_profile, SUBCRITERIA as A4_CRIT
    from modules.agents.agent5_prevention import build_agent5_profile, SUBCRITERIA as A5_CRIT
    from modules.agents.agent6_equity import build_agent6_profile, SUBCRITERIA as A6_CRIT
    from modules.agents.agent7_outbreak import build_agent7_profile
    from modules.agents.agent8_social import build_agent8_profile

    agents = [
        ("Agent 1 (Epidemiology)", build_agent1_profile, A1_CRIT),
        ("Agent 2 (Transmission)", build_agent2_profile, A2_CRIT),
        ("Agent 3 (Healthcare Impact)", build_agent3_profile, A3_CRIT),
        ("Agent 4 (Clinical Urgency)", build_agent4_profile, A4_CRIT),
        ("Agent 5 (Prevention)", build_agent5_profile, A5_CRIT),
        ("Agent 6 (Equity)", build_agent6_profile, A6_CRIT),
    ]

    for name, builder, subcrits in agents:
        print(f"\n  --- {name} ---")
        for sc in subcrits:
            cid = sc["id"]
            if cid in df_wide.columns:
                print(f"    ✅ MATCH: '{cid}'")
            else:
                print(f"    ❌ KEY MISMATCH: '{cid}' is NOT in df_wide!")
        
        prof = builder(df_wide)
        print(f"\n    Profile Matrix Output ({name}):")
        print(prof.to_string(index=False))

    print(f"\n  --- Agent 7 (Outbreak Dynamics) ---")
    a7_prof = build_agent7_profile(df_wide)
    print(a7_prof.to_string(index=False))

    print(f"\n  --- Agent 8 (Social & Economic) ---")
    a8_prof = build_agent8_profile(df_wide)
    print(a8_prof.to_string(index=False))

except Exception as e:
    print(f"  ❌ Agent testing encountered an error: {e}")

print("\n" + "=" * 80)
print("  DIAGNOSTIC TEST COMPLETE")
print("=" * 80)