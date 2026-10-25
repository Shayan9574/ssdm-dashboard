"""Storage, access, and cache keys."""
import json
import sys
import types

import pytest

from modules import state


@pytest.fixture
def json_store(tmp_path, monkeypatch):
    b = state.JSONBackend(tmp_path / "inputs.json")
    state.set_backend(b)
    state._GLOBAL_SESSION.clear()
    yield b
    state.set_backend(state.MemoryBackend())


def test_put_get_persist_and_log(json_store, tmp_path):
    state.put("weights", "A1", {"values": {"x": 1.0}}, provenance="direct selection")
    assert state.get("weights", "A1")["provenance"] == "direct selection"
    on_disk = json.loads((tmp_path / "inputs.json").read_text())
    assert on_disk["weights"]["A1"]["values"] == {"x": 1.0}
    log = (tmp_path / "inputs_log.jsonl").read_text().strip().splitlines()
    assert len(log) == 1 and json.loads(log[0])["key"] == "A1"
    state.delete("weights", "A1")
    assert state.get("weights", "A1") is None


def test_context_key_changes_with_every_input(json_store):
    k0 = state.context_key()
    state.put("intensity", "S4", {"expert_respondents": {"a": {"severity": "Negligible", "impact": "Negligible"}}})
    k1 = state.context_key()
    state.set_session_setting("attainment_mode", "binary")
    k2 = state.context_key()
    state.clear_session_settings()
    assert len({k0, k1, k2}) == 3 and state.context_key() == k1


def test_cached_functions_recompute_after_an_input_changes(json_store):
    calls = []

    @state.cached
    def f(x):
        calls.append(x)
        return state.get("weights", "A2", {}).get("values", {}).get("w", 0) + x

    assert f(1) == 1 and f(1) == 1 and len(calls) == 1
    state.put("weights", "A2", {"values": {"w": 5}}, provenance="direct selection")
    assert f(1) == 6 and len(calls) == 2


def test_editing_requires_the_password(json_store, monkeypatch):
    monkeypatch.setenv("EDITOR_PASSWORD", "secret")
    monkeypatch.delenv("SSDM_EDITOR", raising=False)
    with pytest.raises(PermissionError):
        state.put("weights", "A3", {"values": {}})
    monkeypatch.setenv("SSDM_EDITOR", "1")
    state.put("weights", "A3", {"values": {}})


def test_sheet_backend_round_trip(monkeypatch):
    rows = {"inputs": [], "log": [], "ledger": []}

    class WS:
        def __init__(self, name): self.name = name
        def append_row(self, r): rows[self.name].append(r)
        def append_rows(self, rs): rows[self.name].extend(rs)
        def clear(self): rows[self.name] = []
        def update(self, values, range_name): rows[self.name] = list(values)
        def get_all_records(self):
            data = rows[self.name]
            if not data: return []
            hdr = data[0]
            return [dict(zip(hdr, r)) for r in data[1:]]

    class Book:
        def worksheet(self, t):
            if not rows[t]: raise KeyError(t)
            return WS(t)
        def add_worksheet(self, title, rows, cols): return WS(title)

    fake = types.SimpleNamespace(authorize=lambda c: types.SimpleNamespace(open_by_key=lambda k: Book()))
    monkeypatch.setitem(sys.modules, "gspread", fake)
    import google.oauth2.service_account as sa
    monkeypatch.setattr(sa.Credentials, "from_service_account_info", staticmethod(lambda i, scopes: object()))
    b = state.SheetBackend("sheet", {"x": 1})
    state.set_backend(b)
    state.put("limits", "A1", {"values": {"c": 2.0}}, provenance="expert")
    state.append_ledger([{"disease": "x", "status": "validated"}])
    b2 = state.SheetBackend("sheet", {"x": 1})
    doc = b2.load()
    assert doc["limits"]["A1"]["values"] == {"c": 2.0}
    assert doc["ledger"] == [{"disease": "x", "status": "validated"}]
    assert any(r[1] == "limits" for r in rows["log"][1:])
    state.set_backend(state.MemoryBackend())
