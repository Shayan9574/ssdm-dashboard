import json
import os
import re
import requests

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
KNOWN_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]

def normalize_model(name: str) -> str:
    """'Gemini 3.1 Flash' -> 'gemini-3.1-flash'; empty -> default."""
    n = str(name or "").strip().lower().replace(" ", "-")
    return n or DEFAULT_GEMINI_MODEL

def _scrub(msg: str) -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    m = str(msg)
    if key:
        m = m.replace(key, "***")
    return re.sub(r"key=[^&\s]+", "key=***", m)

def gemini_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))

def _post_json(url, headers, payload, timeout=150):
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def call_openai_chat(api_key: str, model: str, prompt: str, temperature: float = 0.2):
    if not api_key:
        raise ValueError("Missing OpenAI API key.")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": float(temperature),
        "messages": [
            {"role": "system", "content": "You are a public-health decision support analyst. Write clearly for decision-makers."},
            {"role": "user", "content": prompt},
        ],
    }
    data = _post_json(url, headers, payload, timeout=90)
    return data["choices"][0]["message"]["content"]

def call_gemini(api_key: str, model: str, prompt: str, temperature: float = 0.2):
    model = normalize_model(model)
    if not api_key:
        raise ValueError("Missing Gemini API key.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    payload = {
        "generationConfig": {"temperature": float(temperature)},
        "contents": [{"parts": [{"text": prompt}]}],
    }
    data = _post_json(url, headers, payload, timeout=90)

    cands = data.get("candidates", [])
    if not cands:
        return ""
    parts = cands[0].get("content", {}).get("parts", [])
    txt = [p["text"] for p in parts if "text" in p]
    return "\n".join(txt).strip()

def call_gemini_grounded(api_key: str, model: str, prompt: str,
                         temperature: float = 0.3):
    """Gemini with Google Search grounding: returns (text, sources), where
    sources is a list of {title, uri} drawn from the grounding metadata."""
    if not api_key:
        raise ValueError("Missing Gemini API key.")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": temperature},
    }
    import time as _time
    data, last_err = None, None
    for attempt in range(3):
        try:
            data = _post_json(url, {"Content-Type": "application/json", "x-goog-api-key": api_key},
                              payload, timeout=240)
            break
        except Exception as e:
            last_err = e
            _time.sleep(5 * (attempt + 1))
    if data is None:
        raise last_err
    cand = (data.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "")
                   for p in cand.get("content", {}).get("parts", []))
    sources = []
    for ch in cand.get("groundingMetadata", {}).get("groundingChunks", []):
        web = ch.get("web", {})
        if web.get("uri"):
            sources.append({"title": web.get("title", web["uri"]),
                            "uri": web["uri"]})
    return text.strip(), sources


def _with_fallback(fn, api_key, model, *a, **kw):
    try:
        return fn(api_key, normalize_model(model), *a, **kw)
    except Exception as e1:
        m2 = DEFAULT_GEMINI_MODEL
        if normalize_model(model) != m2:
            try:
                return fn(api_key, m2, *a, **kw)
            except Exception as e2:
                raise RuntimeError(_scrub(f"{e2} (after falling back from "
                                          f"'{model}' to {m2})")) from None
        raise RuntimeError(_scrub(str(e1))) from None
