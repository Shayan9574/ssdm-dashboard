"""
state.py

Every judgment input and method setting of the dashboard, in one store:

  weights      scope -> entry   (agent baseline "A1", scenario specific
                                 "A1::S3", agent level "agent_level::global")
  limits       agent -> entry   (expert acceptable limits)
  respondents  scope -> {respondent -> {item -> linguistic term}}
  intensity    scenario -> {"expert": {...}, "ai": {...}}
  settings     name -> value    (method configuration saved by the editor)
  ledger       append only list of external evidence records

Entries carry their provenance and the time and author of the change.

Backends, chosen automatically:
  * Google Sheet, when the deployment secrets hold a service account
    (gcp_service_account) and SSDM_SHEET_ID: shared by every viewer and
    persistent across restarts; every change is also appended to a log tab;
  * a local JSON file in the data folder (Colab, local runs);
  * memory only, when neither is writable.

Viewers may move analysis parameters within their own session; those
session values never reach the store. Only the editor (password) saves.

context_key() hashes the complete effective state, the effective settings
and the feed files; every cached computation is keyed on it, so a change
of any input invalidates exactly the affected results, and two users with
different inputs never share a cached result.
"""

import copy
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

KINDS = ("weights", "limits", "respondents", "intensity", "settings", "ledger")

DEFAULT_SETTINGS: Dict[str, Any] = {
    "attainment_mode": "graded_fixed",
    "theta_fixed": 0.3,
    "theta_min": 0.3,
    "theta_max": 0.7,
    "dominance_guard": True,
    "guard_min_shared": 1,
    "fuzzy_distance": False,
    "history_window": "post_phe_rolling5",
    "scenario_tightening": True,
    "tightening_quantile": 0.90,
    "emphasis_source": "data",          # data | registry_k3 (ablation only)
    "frequency_source": "computed",     # computed | registry
    "episode_gap_weeks": 3,
    "episode_min_weeks": 2,
    "aggregation_rule": "mean",         # mean | minmax_geometric
    "lambda": 0.7, "alpha": 1.0, "beta": 1.0, "gamma": 1.0,
    "theta_si": 0.7,
    "gemini_model": "gemini-2.5-flash",
}

_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _secret(name: str, default=None):
    v = os.environ.get(name)
    if v:
        return v
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


# ----------------------------------------------------------------------
# backends
# ----------------------------------------------------------------------

class MemoryBackend:
    name = "memory (not saved)"

    def __init__(self):
        self.doc = {k: ({} if k != "ledger" else []) for k in KINDS}

    def load(self) -> dict:
        return copy.deepcopy(self.doc)

    def save(self, doc: dict, change: Optional[dict] = None) -> None:
        self.doc = copy.deepcopy(doc)


class JSONBackend(MemoryBackend):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.name = f"local file ({path.name})"

    def load(self) -> dict:
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text())
                return {k: d.get(k, {} if k != "ledger" else []) for k in KINDS}
            except Exception:
                pass
        return super().load()

    def save(self, doc: dict, change: Optional[dict] = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1, default=str))
        os.replace(tmp, self.path)
        if change:
            log = self.path.with_name(self.path.stem + "_log.jsonl")
            with open(log, "a") as f:
                f.write(json.dumps(change, default=str) + "\n")


class SheetBackend(MemoryBackend):
    """One row per (kind, key) in the 'inputs' tab, JSON valued; every change
    appended to the 'log' tab."""

    def __init__(self, sheet_id: str, sa_info: dict):
        super().__init__()
        import gspread
        from google.oauth2.service_account import Credentials
        creds = Credentials.from_service_account_info(
            sa_info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        self.book = gspread.authorize(creds).open_by_key(sheet_id)
        self.name = "Google Sheet"
        self.inputs = self._tab("inputs", ["kind", "key", "value", "updated", "by"])
        self.log = self._tab("log", ["time", "kind", "key", "value", "by", "action"])
        self.ledger_tab = self._tab("ledger", ["record"])

    def _tab(self, title, header):
        try:
            ws = self.book.worksheet(title)
        except Exception:
            ws = self.book.add_worksheet(title=title, rows=1000, cols=len(header))
            ws.append_row(header)
        return ws

    def load(self) -> dict:
        doc = super().load()
        for r in self.inputs.get_all_records():
            k, key = r.get("kind"), str(r.get("key"))
            if k not in KINDS:
                continue
            try:
                val = json.loads(r.get("value") or "null")
            except Exception:
                continue
            if k != "ledger":
                doc[k][key] = val
        for r in self.ledger_tab.get_all_records():
            try:
                doc["ledger"].append(json.loads(r.get("record") or "{}"))
            except Exception:
                continue
        return doc

    def append_ledger(self, rows: list) -> None:
        self.ledger_tab.append_rows([[json.dumps(r, default=str)[:45000]] for r in rows])

    def save(self, doc: dict, change: Optional[dict] = None) -> None:
        rows = [["kind", "key", "value", "updated", "by"]]
        for k in KINDS:
            if k == "ledger":
                continue
            for key, val in doc[k].items():
                rows.append([k, key, json.dumps(val, default=str), _now(), ""])
        self.inputs.clear()
        self.inputs.update(values=rows, range_name="A1")
        if change:
            self.log.append_row([change.get("time"), change.get("kind"), change.get("key"),
                                 json.dumps(change.get("value"), default=str)[:45000],
                                 change.get("by", ""), change.get("action", "set")])


_BACKEND = None


def backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    sid = _secret("SSDM_SHEET_ID")
    sa = _secret("gcp_service_account")
    if sid and sa:
        try:
            info = json.loads(sa) if isinstance(sa, str) else dict(sa)
            _BACKEND = SheetBackend(str(sid), info)
            return _BACKEND
        except Exception as e:  # fall back, but say why
            os.environ["SSDM_STORE_ERROR"] = f"Google Sheet unavailable: {e}"
    from modules.feeds import data_dir
    p = Path(_secret("SSDM_STATE_FILE", str(data_dir() / "ssdm_inputs.json")))
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        test = p.with_suffix(".write_test")
        test.write_text("ok")
        test.unlink()
        _BACKEND = JSONBackend(p)
    except Exception:
        _BACKEND = MemoryBackend()
    return _BACKEND


def set_backend(b) -> None:
    """Tests and scripts select a backend explicitly."""
    global _BACKEND, _SHARED, _SHARED_AT
    _BACKEND = b
    _SHARED, _SHARED_AT = None, 0.0


# ----------------------------------------------------------------------
# shared document and session view
# ----------------------------------------------------------------------

_SHARED: Optional[dict] = None
_SHARED_AT = 0.0
SHARED_TTL = 60.0


def _shared() -> dict:
    global _SHARED, _SHARED_AT
    with _LOCK:
        if _SHARED is None or time.time() - _SHARED_AT > SHARED_TTL:
            _SHARED = backend().load()
            _SHARED_AT = time.time()
        return _SHARED


def _session() -> dict:
    """Per viewer session overrides (analysis parameters only)."""
    try:
        import streamlit as st
        return st.session_state.setdefault("_ssdm_session_settings", {})
    except Exception:
        return _GLOBAL_SESSION


_GLOBAL_SESSION: Dict[str, Any] = {}


def get(kind: str, key: str, default=None):
    return copy.deepcopy(_shared()[kind].get(key, default))


def items(kind: str) -> dict:
    return copy.deepcopy(_shared()[kind])


def put(kind: str, key: str, value, provenance: Optional[str] = None, by: str = "") -> None:
    if not can_edit():
        raise PermissionError("Editing requires the editor password.")
    global _SHARED_AT
    doc = copy.deepcopy(_shared())
    if provenance is not None and isinstance(value, dict):
        value = {**value, "provenance": provenance, "updated": _now(), "by": by or editor_name()}
    doc[kind][key] = value
    with _LOCK:
        backend().save(doc, {"time": _now(), "kind": kind, "key": key, "value": value,
                             "by": by or editor_name(), "action": "set"})
        _SHARED.clear(); _SHARED.update(doc)
        _SHARED_AT = time.time()


def delete(kind: str, key: str) -> None:
    if not can_edit():
        raise PermissionError("Editing requires the editor password.")
    global _SHARED_AT
    doc = copy.deepcopy(_shared())
    doc[kind].pop(key, None)
    with _LOCK:
        backend().save(doc, {"time": _now(), "kind": kind, "key": key, "value": None,
                             "by": editor_name(), "action": "delete"})
        _SHARED.clear(); _SHARED.update(doc)
        _SHARED_AT = time.time()


def append_ledger(rows: list) -> int:
    if not can_edit():
        raise PermissionError("Editing requires the editor password.")
    global _SHARED_AT
    doc = copy.deepcopy(_shared())
    doc["ledger"] = list(doc.get("ledger", [])) + list(rows)
    with _LOCK:
        b = backend()
        if hasattr(b, "append_ledger"):
            b.append_ledger(rows)
            b.log.append_row([_now(), "ledger", "append", f"{len(rows)} rows", editor_name(), "append"])
        else:
            b.save(doc, {"time": _now(), "kind": "ledger", "key": "append",
                         "value": f"{len(rows)} rows", "by": editor_name(), "action": "append"})
        _SHARED.clear(); _SHARED.update(doc)
        _SHARED_AT = time.time()
    return len(rows)


def ledger() -> list:
    return copy.deepcopy(_shared().get("ledger", []))


# ----------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------

def setting(name: str):
    s = _session()
    if name in s:
        return s[name]
    saved = _shared()["settings"]
    return saved.get(name, DEFAULT_SETTINGS.get(name))


def settings() -> Dict[str, Any]:
    return {k: setting(k) for k in DEFAULT_SETTINGS}


def set_session_setting(name: str, value) -> None:
    _session()[name] = value


def clear_session_settings() -> None:
    _session().clear()


def save_settings_as_default() -> None:
    for k, v in settings().items():
        if _shared()["settings"].get(k) != v:
            put("settings", k, v)
    clear_session_settings()


# ----------------------------------------------------------------------
# access
# ----------------------------------------------------------------------

def password_configured() -> bool:
    return bool(_secret("EDITOR_PASSWORD"))


def can_edit() -> bool:
    """True without a configured password (local use), for scripts run by
    the deployer (SSDM_EDITOR=1 in their environment), and for a session
    signed in with the editor password."""
    if not password_configured() or os.environ.get("SSDM_EDITOR") == "1":
        return True
    try:
        import streamlit as st
        return bool(st.session_state.get("_ssdm_editor", False))
    except Exception:
        return os.environ.get("SSDM_EDITOR") == "1"


def login(password: str, name: str = "") -> bool:
    import hmac
    ok = bool(password) and hmac.compare_digest(str(password), str(_secret("EDITOR_PASSWORD", "")))
    if ok:
        import streamlit as st
        st.session_state["_ssdm_editor"] = True
        st.session_state["_ssdm_editor_name"] = name or "editor"
    return ok


def logout() -> None:
    import streamlit as st
    st.session_state["_ssdm_editor"] = False


def editor_name() -> str:
    try:
        import streamlit as st
        return st.session_state.get("_ssdm_editor_name", "editor")
    except Exception:
        return "script"


# ----------------------------------------------------------------------
# cache key
# ----------------------------------------------------------------------

def context_key() -> str:
    from modules import feeds, live_connectors as lc
    doc = _shared()
    payload = json.dumps({"doc": {k: doc[k] for k in KINDS if k != "ledger"},
                          "settings": settings(), "feeds": feeds.feed_signature(),
                          "as_of": str(lc.AS_OF)}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def cached(fn):
    """Caches fn on its arguments and the complete context (inputs, settings,
    feeds). Shared across sessions only where the context is identical."""
    import streamlit as st

    @st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=512)
    def _inner(ctx: str, name: str, args: tuple, kwargs: tuple):
        return fn(*args, **dict(kwargs))

    def wrapper(*args, **kwargs):
        return _inner(context_key(), fn.__name__, args, tuple(sorted(kwargs.items())))

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    wrapper.uncached = fn
    wrapper.clear = _inner.clear
    return wrapper
