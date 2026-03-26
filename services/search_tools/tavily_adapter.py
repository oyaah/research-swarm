from __future__ import annotations

import time
from typing import Any

import requests


def web_search_tavily(api_key: str, query: str, num: int = 5) -> list[dict[str, Any]]:
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": num,
        "include_raw_content": False,
    }
    res = requests.post(url, json=payload, timeout=20)
    res.raise_for_status()
    data = res.json()
    now = str(int(time.time()))
    results = []
    for row in data.get("results", [])[:num]:
        results.append(
            {
                "title": row.get("title", "untitled"),
                "url": row.get("url", ""),
                "snippet": row.get("content", ""),
                "timestamp": now,
            }
        )
    return results
