from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
import re
from typing import Any
import base64
from urllib.parse import parse_qs, unquote, urlparse

import requests

from services.qdrant_client import qdrant_store
from services.swarm_engine.artifact_exports import export_chart, export_pdf
from services.swarm_engine.settings import settings
from services.search_tools.searchapiio_adapter import web_search_searchapiio
from services.search_tools.web_fallback_adapter import web_search_duckduckgo
from services.search_tools.tavily_adapter import web_search_tavily
from tools.contracts import tool_result


@lru_cache(maxsize=1)
def playwright_available() -> bool:
    """Check if playwright and chromium are installed. Result is cached after first call."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        return True
    except Exception:
        return False


def _normalized_key(value: str) -> str:
    v = (value or "").strip()
    if v.startswith("${") and v.endswith("}"):
        v = v[2:-1].strip()
    return v


def _decode_duckduckgo_href(href: str) -> str:
    if not href:
        return ""
    if "duckduckgo.com/l/?" in href:
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        uddg = qs.get("uddg", [""])[0]
        if uddg:
            return unquote(uddg)
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/l/?"):
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        uddg = qs.get("uddg", [""])[0]
        if uddg:
            return unquote(uddg)
    return ""


def _decode_bing_href(href: str) -> str:
    if not href:
        return ""
    if "bing.com/ck/a" not in href:
        return href
    try:
        qs = parse_qs(urlparse(href).query)
        u = qs.get("u", [""])[0]
        if not u:
            return href
        if u.startswith("a1"):
            b64 = u[2:]
            pad = "=" * (-len(b64) % 4)
            decoded = base64.urlsafe_b64decode((b64 + pad).encode("utf-8")).decode("utf-8", errors="ignore")
            if decoded.startswith("http://") or decoded.startswith("https://"):
                return decoded
        try:
            res = requests.get(href, timeout=10, allow_redirects=True)
            if res.url:
                return res.url
        except Exception:
            pass
        return href
    except Exception:
        return href


def web_search_playwright(query: str, num: int = 5, timeout_ms: int = 15000) -> list[dict[str, Any]]:
    """Free search provider: uses requests for URL discovery (reliable across geolocations)
    and playwright_fetch for JS-heavy content extraction downstream.
    Requires playwright to be installed (checked by playwright_available())."""
    if not playwright_available():
        raise RuntimeError("playwright unavailable")
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
    ts = str(int(datetime.now(UTC).timestamp()))
    out: list[dict[str, Any]] = []

    # Strategy 1: DuckDuckGo HTML (best free source, works with requests)
    ddg_urls = [
        "https://duckduckgo.com/html/",
        "https://html.duckduckgo.com/html/",
    ]
    for base in ddg_urls:
        try:
            resp = requests.get(base, params={"q": query}, headers=headers, timeout=20)
            resp.raise_for_status()
            html = resp.text
            # Extract results with snippets
            for match in re.finditer(
                r'<div[^>]*class="[^"]*result\b[^"]*"[^>]*>.*?'
                r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
                r'(?:.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>)?',
                html, flags=re.IGNORECASE | re.DOTALL,
            ):
                href, raw_title, raw_snippet = match.group(1), match.group(2), match.group(3) or ""
                url = _decode_duckduckgo_href(href)
                title = re.sub(r"<[^>]+>", "", raw_title).strip()
                snippet = re.sub(r"<[^>]+>", "", raw_snippet).strip()
                if url and title:
                    out.append({"title": title, "url": url, "snippet": snippet, "timestamp": ts})
                if len(out) >= max(1, int(num)):
                    break
            if out:
                return out
        except Exception:
            continue

    # Strategy 2: Bing via requests
    try:
        resp = requests.get(
            "https://www.bing.com/search",
            params={"q": query, "count": max(1, int(num))},
            headers=headers,
            timeout=20,
        )
        resp.raise_for_status()
        html = resp.text
        for match in re.finditer(
            r'<li[^>]*class="[^"]*\bb_algo\b[^"]*"[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            html, flags=re.IGNORECASE | re.DOTALL,
        ):
            href, raw_title = match.group(1), match.group(2)
            title = re.sub(r"<[^>]+>", "", raw_title).strip()
            if href and title:
                out.append({"title": title, "url": _decode_bing_href(href), "snippet": "", "timestamp": ts})
            if len(out) >= max(1, int(num)):
                break
        if out:
            return out
    except Exception:
        pass

    if not out:
        raise RuntimeError("playwright search returned zero results from all sources")
    return out


def web_search_wikipedia(query: str, num: int = 5) -> list[dict[str, Any]]:
    headers = {
        "User-Agent": "research-swarm/0.1 (+https://localhost; contact=local-dev)",
        "Accept": "application/json",
    }
    res = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max(1, int(num)),
        },
        headers=headers,
        timeout=20,
    )
    res.raise_for_status()
    hits = res.json().get("query", {}).get("search", []) or []
    out: list[dict[str, Any]] = []
    ts = str(int(datetime.now(UTC).timestamp()))
    for h in hits[: max(1, int(num))]:
        title = str(h.get("title", "")).strip()
        if not title:
            continue
        snippet = re.sub(r"<[^>]+>", " ", str(h.get("snippet", ""))).strip()
        url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
        out.append({"title": title, "url": url, "snippet": snippet, "timestamp": ts})
    if not out:
        raise RuntimeError("wikipedia fallback returned zero results")
    return out


def web_search(
    query: str,
    num: int,
    searchapi_api_key: str,
    tavily_api_key: str,
    use_mock: bool,
) -> dict[str, Any]:
    live_query = f"{query} {settings.search_query_suffix}".strip() if settings.search_query_suffix else query
    provider_order = [x.strip().lower() for x in settings.search_provider_order.split(",") if x.strip()]
    pw_aliases = {"playwright", "pw"}
    if playwright_available():
        pw_in_order = [p for p in provider_order if p in pw_aliases]
        if pw_in_order:
            provider_order = pw_in_order + [p for p in provider_order if p not in pw_aliases]
        else:
            provider_order = ["playwright"] + provider_order
    else:
        provider_order = [p for p in provider_order if p not in pw_aliases]
    errors: list[str] = []
    searchapi_api_key = _normalized_key(searchapi_api_key)
    tavily_api_key = _normalized_key(tavily_api_key)
    strip_sites = re.compile(r"\bsite:[^\s]+", flags=re.IGNORECASE)

    def _run_provider(provider: str, search_query: str) -> list[dict[str, Any]]:
        if provider == "tavily":
            if not tavily_api_key:
                raise RuntimeError("tavily key missing")
            return web_search_tavily(tavily_api_key, search_query, num)
        if provider in {"searchapiio", "searchapi"}:
            if not searchapi_api_key or searchapi_api_key == "stub":
                raise RuntimeError("searchapi.io key missing")
            return web_search_searchapiio(searchapi_api_key, search_query, num)
        if provider in {"duckduckgo", "ddg"}:
            return web_search_duckduckgo(search_query, num)
        if provider in {"playwright", "pw"}:
            return web_search_playwright(search_query, num=num, timeout_ms=max(5000, int(settings.search_playwright_timeout_ms)))
        raise RuntimeError(f"unknown provider: {provider}")

    def _run_chain(search_query: str) -> tuple[list[dict[str, Any]], str, list[str], list[str]]:
        chain_errors: list[str] = []
        tried: list[str] = []
        for provider in provider_order:
            tried.append(provider)
            try:
                return _run_provider(provider, search_query), provider, tried, chain_errors
            except Exception as exc:
                chain_errors.append(f"{provider}: {exc}")
        return [], "", tried, chain_errors

    results, provider, tried, chain_errors = _run_chain(live_query)
    errors.extend(chain_errors)
    if results:
        return tool_result("ok", {"results": results, "provider": provider})

    relaxed_query = re.sub(r"\s+", " ", strip_sites.sub(" ", live_query)).strip()
    if relaxed_query and relaxed_query != live_query:
        relaxed_results, relaxed_provider, relaxed_tried, relaxed_errors = _run_chain(relaxed_query)
        errors.extend([f"relaxed:{e}" for e in relaxed_errors])
        tried.extend(relaxed_tried)
        if relaxed_results:
            return tool_result(
                "degraded",
                {
                    "results": relaxed_results,
                    "provider": relaxed_provider,
                    "relaxed_query": relaxed_query,
                    "errors": errors,
                },
            )

    if settings.search_allow_final_duckduckgo and "duckduckgo" not in tried and "ddg" not in tried:
        try:
            results = web_search_duckduckgo(live_query, num)
            return tool_result("degraded", {"results": results, "provider": "duckduckgo", "errors": errors})
        except Exception as exc:
            errors.append(f"duckduckgo: {exc}")
    if settings.search_allow_wikipedia_fallback:
        try:
            results = web_search_wikipedia(live_query, num)
            return tool_result("degraded", {"results": results, "provider": "wikipedia", "errors": errors})
        except Exception as exc:
            errors.append(f"wikipedia: {exc}")
    return tool_result("error", {"results": [], "errors": errors, "provider_order": provider_order})


def open_url(url: str, use_mock: bool) -> dict[str, Any]:
    try:
        if use_mock:
            text = f"Mock content for {url}. This document provides contextual evidence and reproducible summary text."
        else:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            raw = response.text[:15000]
            text = re.sub(r"<script.*?>.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()[:8000]
        return tool_result("ok", {"url": url, "text": text})
    except Exception as exc:
        return tool_result("error", {"url": url, "text": "", "error": str(exc)})


def wikipedia_lookup(query: str) -> dict[str, Any]:
    try:
        search_res = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1,
            },
            timeout=15,
        )
        search_res.raise_for_status()
        hits = search_res.json().get("query", {}).get("search", [])
        if not hits:
            return tool_result("error", {"query": query, "text": "", "error": "no wikipedia hit"})
        title = hits[0]["title"]
        page_url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
        sum_res = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}",
            timeout=15,
        )
        sum_res.raise_for_status()
        extract = sum_res.json().get("extract", "")
        return tool_result("ok", {"query": query, "title": title, "url": page_url, "text": extract})
    except Exception as exc:
        return tool_result("error", {"query": query, "text": "", "error": str(exc)})


def playwright_fetch(url: str, timeout_ms: int = 15000) -> dict[str, Any]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        return tool_result("error", {"url": url, "text": "", "error": f"playwright unavailable: {exc}"})

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            # Wait briefly for JS-rendered content to appear
            try:
                page.wait_for_load_state("networkidle", timeout=min(5000, timeout_ms // 2))
            except Exception:
                pass
            # Extract structured content: article/main first, then body
            text = page.evaluate("""() => {
                const sel = document.querySelector('article') || document.querySelector('main') || document.querySelector('[role="main"]');
                if (sel && sel.innerText.trim().length > 200) return sel.innerText.trim();
                return document.body.innerText.trim();
            }""")
            text = (text or "")[:10000]
            browser.close()
        return tool_result("ok", {"url": url, "text": text})
    except Exception as exc:
        return tool_result("error", {"url": url, "text": "", "error": str(exc)})


def summarize_text(text: str) -> dict[str, Any]:
    words = text.split()
    summary = " ".join(words[:60])
    claims = ["Claim: " + " ".join(words[:12]) if words else "Claim: no content"]
    return tool_result("ok", {"summary": summary, "claims": claims})


def qdrant_upsert(document: dict[str, Any]) -> dict[str, Any]:
    doc_id = qdrant_store.upsert(document)
    return tool_result("ok", {"id": doc_id})


def qdrant_search(session_id: str, query: str, limit: int = 5) -> dict[str, Any]:
    results = qdrant_store.search(session_id=session_id, query=query, limit=limit)
    return tool_result("ok", {"results": results})


def snapshot_url(url: str, session_id: str) -> dict[str, Any]:
    snap_dir = Path("artifacts/snapshots") / session_id
    snap_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snap_dir / f"{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}.txt"
    snapshot_path.write_text(f"Snapshot for {url} @ {datetime.now(UTC).isoformat()}", encoding="utf-8")
    return tool_result("ok", {"path": str(snapshot_path), "url": url})


def ask_user_question(question: str) -> dict[str, Any]:
    return tool_result("ok", {"question": question, "requires_human": True})


def export_pdf_tool(session_id: str, markdown: str) -> dict[str, Any]:
    report_dir = Path("artifacts/reports") / session_id
    report_dir.mkdir(parents=True, exist_ok=True)
    path, warning = export_pdf(report_dir, markdown)
    if path:
        return tool_result("ok", {"path": path, "warning": warning or ""})
    return tool_result("error", {"path": "", "warning": warning or "pdf export failed"})


def plot_chart_tool(session_id: str, markdown: str) -> dict[str, Any]:
    report_dir = Path("artifacts/reports") / session_id
    report_dir.mkdir(parents=True, exist_ok=True)
    path, warning = export_chart(report_dir, markdown)
    if path:
        return tool_result("ok", {"path": path, "warning": warning or ""})
    return tool_result("error", {"path": "", "warning": warning or "chart export failed"})


def schema() -> list[dict[str, Any]]:
    return [
        {
            "name": "web_search",
            "description": "Search the web and return top N results with snippet and url",
            "inputs": {"q": "string", "num": "integer"},
            "outputs": {"results": [{"title": "string", "url": "string", "snippet": "string", "timestamp": "string"}]},
        },
        {
            "name": "open_url",
            "description": "Open URL and return page text",
            "inputs": {"url": "string"},
            "outputs": {"url": "string", "text": "string"},
        },
        {
            "name": "playwright_fetch",
            "description": "Fetch page text with a headless browser for JS-heavy pages",
            "inputs": {"url": "string", "timeout_ms": "integer"},
            "outputs": {"url": "string", "text": "string"},
        },
        {
            "name": "summarize_text",
            "description": "Summarize raw text and extract candidate claims",
            "inputs": {"text": "string"},
            "outputs": {"summary": "string", "claims": ["string"]},
        },
        {
            "name": "wikipedia_lookup",
            "description": "Find a relevant Wikipedia page and return summary text",
            "inputs": {"query": "string"},
            "outputs": {"query": "string", "title": "string", "url": "string", "text": "string"},
        },
        {
            "name": "qdrant_upsert",
            "description": "Store evidence chunk in vector store",
            "inputs": {"document": "object"},
            "outputs": {"id": "string"},
        },
        {
            "name": "qdrant_search",
            "description": "Search vector store by session",
            "inputs": {"session_id": "string", "query": "string", "limit": "integer"},
            "outputs": {"results": ["object"]},
        },
        {
            "name": "snapshot_url",
            "description": "Persist URL snapshot to local artifact store",
            "inputs": {"url": "string", "session_id": "string"},
            "outputs": {"path": "string", "url": "string"},
        },
        {
            "name": "ask_user_question",
            "description": "Emit a human-in-the-loop blocking question",
            "inputs": {"question": "string"},
            "outputs": {"question": "string", "requires_human": "boolean"},
        },
        {
            "name": "export_pdf",
            "description": "Export markdown report to PDF artifact",
            "inputs": {"session_id": "string", "markdown": "string"},
            "outputs": {"path": "string", "warning": "string"},
        },
        {
            "name": "plot_chart",
            "description": "Render a chart image from report markdown",
            "inputs": {"session_id": "string", "markdown": "string"},
            "outputs": {"path": "string", "warning": "string"},
        },
    ]


def tool_registry() -> list[dict[str, Any]]:
    return [
        {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
        }
        for t in schema()
    ]
