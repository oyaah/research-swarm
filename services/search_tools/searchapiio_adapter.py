from __future__ import annotations

import time
from typing import Any

import requests


def web_search_searchapiio(api_key: str, query: str, num: int = 5) -> list[dict[str, Any]]:
    url = "https://www.searchapi.io/api/v1/search"
    base_params = {
        "engine": "google",
        "q": query,
        "num": max(1, int(num)),
    }
    response = requests.get(url, params={**base_params, "api_key": api_key}, timeout=20)
    if response.status_code in {401, 403}:
        response = requests.get(
            url,
            params=base_params,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20,
        )
    response.raise_for_status()
    data = response.json()
    organic = data.get("organic_results", []) or data.get("organic", [])
    now = str(int(time.time()))
    out: list[dict[str, Any]] = []
    for row in organic[: max(1, int(num))]:
        out.append(
            {
                "title": row.get("title", "untitled"),
                "url": row.get("link", row.get("url", "")),
                "snippet": row.get("snippet", row.get("description", "")),
                "timestamp": now,
            }
        )
    return out
