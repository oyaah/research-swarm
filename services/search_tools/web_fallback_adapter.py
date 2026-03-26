from __future__ import annotations

import base64
import re
import time
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests


def web_search_duckduckgo(query: str, num: int = 5) -> list[dict[str, Any]]:
    def _decode_href(href: str) -> str:
        if "duckduckgo.com/l/?" in href:
            qs = parse_qs(urlparse(href).query)
            uddg = qs.get("uddg", [""])[0]
            if uddg:
                return unquote(uddg)
        if href.startswith("http://") or href.startswith("https://"):
            return href
        if href.startswith("//"):
            return "https:" + href
        if href.startswith("/l/?"):
            qs = parse_qs(urlparse(href).query)
            uddg = qs.get("uddg", [""])[0]
            if uddg:
                return unquote(uddg)
        return ""

    def _decode_bing_href(href: str) -> str:
        if "bing.com/ck/a" not in href:
            return href
        try:
            u = parse_qs(urlparse(href).query).get("u", [""])[0]
            if u.startswith("a1"):
                b64 = u[2:]
                pad = "=" * (-len(b64) % 4)
                dec = base64.urlsafe_b64decode((b64 + pad).encode("utf-8")).decode("utf-8", errors="ignore")
                if dec.startswith("http://") or dec.startswith("https://"):
                    return dec
        except Exception:
            pass
        try:
            resolved = requests.get(href, timeout=10, allow_redirects=True)
            if resolved.url:
                return resolved.url
        except Exception:
            pass
        return href

    def _parse_ddg(html: str) -> list[dict[str, Any]]:
        links = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, flags=re.IGNORECASE)
        out: list[dict[str, Any]] = []
        now = str(int(time.time()))
        for href, title in links:
            clean_title = re.sub(r"<[^>]+>", "", title).strip()
            clean_url = _decode_href(href)
            if not clean_url or not clean_title:
                continue
            out.append({"title": clean_title, "url": clean_url, "snippet": "", "timestamp": now})
            if len(out) >= max(1, int(num)):
                break
        return out

    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
    ddg_urls = [
        "https://duckduckgo.com/html/",
        "https://html.duckduckgo.com/html/",
        "https://lite.duckduckgo.com/lite/",
    ]
    ddg_errors: list[str] = []
    for base in ddg_urls:
        try:
            response = requests.get(base, params={"q": query}, headers=headers, timeout=20)
            response.raise_for_status()
            out = _parse_ddg(response.text)
            if out:
                return out
            ddg_errors.append(f"{base}: zero parsed results")
        except Exception as exc:
            ddg_errors.append(f"{base}: {exc}")

    try:
        bing = requests.get(
            "https://www.bing.com/search",
            params={"q": query, "count": max(1, int(num))},
            headers=headers,
            timeout=20,
        )
        bing.raise_for_status()
        html = bing.text
        rows = re.findall(
            r'<li[^>]*class="[^"]*\bb_algo\b[^"]*"[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        out: list[dict[str, Any]] = []
        now = str(int(time.time()))
        for href, title in rows:
            clean_title = re.sub(r"<[^>]+>", "", title).strip()
            if not href or not clean_title:
                continue
            out.append({"title": clean_title, "url": _decode_bing_href(href), "snippet": "", "timestamp": now})
            if len(out) >= max(1, int(num)):
                break
        if out:
            return out
        raise RuntimeError("bing fallback returned zero parsed results")
    except Exception as exc:
        raise RuntimeError("duckduckgo search failed; bing fallback also failed; " + " | ".join(ddg_errors) + f" | bing: {exc}") from exc
