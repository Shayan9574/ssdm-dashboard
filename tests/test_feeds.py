"""Feed access: the data branch path used by the web deployment, and a
deployment without any feed."""
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from modules import feeds


def test_data_branch_download_and_refresh(tmp_path, monkeypatch):
    branch = tmp_path / "branch"
    (branch / "feeds").mkdir(parents=True)
    df = pd.DataFrame({"weekendingdate": ["2026-10-03"], "jurisdiction": ["USA"],
                       "totalconfc19newadmper100k": ["1.5"]})
    f = branch / "feeds" / "Weekly Hospital Respiratory Dat.parquet"
    df.to_parquet(f, index=False)
    man = {"files": {f.name: {"sha256": hashlib.sha256(f.read_bytes()).hexdigest()}}}
    (branch / "manifest.json").write_text(json.dumps(man))
    calls = []

    def fake_get(path):
        calls.append(path)
        p = branch / path
        return p.read_bytes() if p.exists() else None

    monkeypatch.setattr(feeds, "_remote_get", fake_get)
    monkeypatch.setenv("SSDM_DATA_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("SSDM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SSDM_REMOTE_DATA", "1")
    monkeypatch.setattr(feeds, "workbook_path", lambda: None)
    out = feeds.load_feed("Weekly Hospital Respiratory Dat")
    assert list(out["totalconfc19newadmper100k"]) == ["1.5"]
    assert feeds._source_for("Weekly Hospital Respiratory Dat")[0] == "data branch"
    n = len(calls)
    feeds.load_feed("Weekly Hospital Respiratory Dat")      # within the TTL: no new download
    assert len(calls) == n
    # a new week appears: the changed file is downloaded after the TTL
    df2 = pd.DataFrame({"weekendingdate": ["2026-10-10"], "jurisdiction": ["USA"],
                        "totalconfc19newadmper100k": ["2.0"]})
    df2.to_parquet(f, index=False)
    man = {"files": {f.name: {"sha256": hashlib.sha256(f.read_bytes()).hexdigest()}}}
    (branch / "manifest.json").write_text(json.dumps(man))
    feeds.sync_remote(force=True)
    assert list(feeds.load_feed("Weekly Hospital Respiratory Dat")["totalconfc19newadmper100k"]) == ["2.0"]


def test_missing_feeds_return_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SSDM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SSDM_REMOTE_DATA", "0")
    monkeypatch.setattr(feeds, "workbook_path", lambda: None)
    for s in feeds.FEED_SHEETS:
        assert feeds.load_feed(s).empty
    st = feeds.data_status()
    assert (st["Source"] == "missing").all()
