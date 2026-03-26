"""
Check which LLM and search providers are configured and reachable.
Run from the project root: python scripts/check_providers.py
"""
from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.swarm_engine.settings import settings
from services.swarm_engine.llm import LLMAdapter

PING_PROMPT = "Reply with the single word: PONG"


async def check_llm_providers() -> None:
    llm = LLMAdapter(
        use_mock=False,
        groq_api_key=settings.groq_api_key,
        openai_api_key=settings.openai_api_key,
        anthropic_api_key=settings.anthropic_api_key,
        deepseek_api_key=settings.deepseek_api_key,
        gemini_api_key=settings.gemini_api_key,
        model_groq=settings.model_groq,
        model_groq_backup=settings.model_groq_backup,
        model_groq_qwen32b=settings.model_groq_qwen32b,
        model_groq_llama_scout=settings.model_groq_llama_scout,
        model_groq_kimi_k2=settings.model_groq_kimi_k2,
        model_openai=settings.model_openai,
        model_anthropic=settings.model_anthropic,
        model_primary=settings.model_primary,
        model_fallback=settings.model_fallback,
    )

    providers = [
        ("Groq", settings.groq_api_key, llm._chat_groq),
        ("OpenAI", settings.openai_api_key, llm._chat_openai),
        ("Anthropic", settings.anthropic_api_key, llm._chat_anthropic),
        ("DeepSeek/Together", settings.deepseek_api_key, llm._chat_deepseek),
        ("Gemini", settings.gemini_api_key, llm._chat_gemini),
    ]

    print("\nLLM Providers:")
    print("─" * 50)
    any_llm_ok = False
    for name, key, fn in providers:
        if not key or key in {"stub", ""}:
            print(f"  {name:<20} [SKIP] no key configured")
            continue
        try:
            result = await asyncio.wait_for(fn(PING_PROMPT, temperature=0.0), timeout=15)
            ok = bool(result and result.strip())
            status = "[OK]  " if ok else "[FAIL] empty response"
            if ok:
                any_llm_ok = True
        except Exception as exc:
            status = f"[FAIL] {str(exc)[:80]}"
        print(f"  {name:<20} {status}")

    if not any_llm_ok:
        print("\n  WARNING: No LLM provider responded. Set at least one key in .env")
    print()


def check_search_providers() -> None:
    from services.search_tools.tavily_adapter import web_search_tavily
    from services.search_tools.searchapiio_adapter import web_search_searchapiio

    providers = [
        ("Tavily", settings.tavily_api_key, lambda: web_search_tavily(settings.tavily_api_key, "test query", 1)),
        ("SearchAPI.io", settings.searchapi_api_key, lambda: web_search_searchapiio(settings.searchapi_api_key, "test query", 1)),
    ]

    print("Search Providers:")
    print("─" * 50)
    any_search_ok = False
    for name, key, fn in providers:
        if not key or key in {"stub", ""}:
            print(f"  {name:<20} [SKIP] no key configured")
            continue
        try:
            results = fn()
            ok = isinstance(results, list) and len(results) > 0
            status = f"[OK]  {len(results)} results" if ok else "[FAIL] empty results"
            if ok:
                any_search_ok = True
        except Exception as exc:
            status = f"[FAIL] {str(exc)[:80]}"
        print(f"  {name:<20} {status}")

    if not any_search_ok:
        print("\n  WARNING: No search provider responded. Set TAVILY_API_KEY or SEARCHAPI_API_KEY in .env")
    print()


def check_infra() -> None:
    import requests

    services = [
        ("Qdrant", "http://localhost:6333/"),
        ("API", "http://localhost:8000/healthz"),
    ]
    print("Local Services:")
    print("─" * 50)
    for name, url in services:
        try:
            r = requests.get(url, timeout=3)
            status = f"[OK]  HTTP {r.status_code}" if r.ok else f"[FAIL] HTTP {r.status_code}"
        except Exception as exc:
            status = f"[DOWN] {str(exc)[:60]}"
        print(f"  {name:<20} {status}")
    print()


async def main() -> None:
    print("Research Swarm — Provider Health Check")
    print("=" * 50)
    check_infra()
    check_search_providers()
    await check_llm_providers()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
