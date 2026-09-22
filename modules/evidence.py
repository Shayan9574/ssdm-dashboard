import requests

def serper_search(api_key: str, query: str, num: int = 5):
    if not api_key:
        raise ValueError("Missing SERPER API key.")

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": int(num)}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    out = []
    for item in data.get("organic", [])[:num]:
        out.append({
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        })
    return out