"""
data_cache.py

Speed layer: every sheet of the master workbook is parsed from Excel
ONCE, stored as Parquet on the fast local disk, and read back in
milliseconds thereafter. The cache invalidates automatically when the
master workbook's modification time changes (a refresh, a sync). In
Colab the cache lives on the virtual machine's local disk, not on the
Drive mount, which is where the original slowness came from.
"""

import hashlib
import os
from pathlib import Path

import pandas as pd


def _cache_dir() -> Path:
    env = os.environ.get("SSDM_CACHE_DIR")
    if env:
        d = Path(env)
    elif Path("/content").exists():
        d = Path("/content/ssdm_cache")
    else:
        d = Path(__file__).resolve().parent.parent / ".cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_excel_sheet(path, sheet_name=0, **kwargs) -> pd.DataFrame:
    """Drop in replacement for pd.read_excel on workbook sheets, backed by
    a Parquet cache keyed on the file's modification time and the read
    arguments. Falls back to a direct Excel read on any cache trouble."""
    p = Path(path)
    try:
        mtime = int(p.stat().st_mtime)
        key = hashlib.md5(
            f"{p}|{sheet_name}|{sorted(kwargs.items())}|{mtime}".encode()
        ).hexdigest()[:20]
        f = _cache_dir() / f"{key}.parquet"
        if f.exists():
            return pd.read_parquet(f)
        df = pd.read_excel(p, sheet_name=sheet_name, **kwargs)
        try:
            out = df.copy()
            out.columns = [str(c) for c in out.columns]
            out.to_parquet(f, index=False)
        except Exception:
            pass
        return df
    except Exception:
        return pd.read_excel(p, sheet_name=sheet_name, **kwargs)
