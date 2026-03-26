from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.swarm_engine.settings import settings


@dataclass
class SwarmConfig:
    task_type: str
    complexity: str
    researchers: int
    verifier_passes: int
    min_verified_evidence: int
    source_hints: list[str]
    model_routes: dict[str, dict[str, Any]]
    researcher_routes: list[dict[str, Any]]


def _groq_route(model: str) -> dict[str, Any]:
    return {"provider": "groq", "model": model, "provider_locked": False}


def _unique_models(models: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for m in models:
        if not m:
            continue
        if m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


def _groq_model_catalog() -> dict[str, str]:
    return {
        "reason_large": settings.model_groq,
        "reason_small": settings.model_groq_backup,
        "reason_mid": settings.model_groq_qwen32b or settings.model_groq_backup,
        "tool_scout": settings.model_groq_llama_scout or settings.model_groq_backup,
        "tool_kimi": settings.model_groq_kimi_k2,
    }


def _build_groq_researcher_routes(task_type: str, complexity: str, researchers: int) -> list[dict[str, Any]]:
    # Spawn mixed-size Groq lanes: tool/browser scouts + deep reasoning lanes.
    cat = _groq_model_catalog()
    small = cat["reason_small"]
    mid = cat["reason_mid"]
    large = cat["reason_large"]
    scout = cat["tool_scout"]
    kimi = cat["tool_kimi"]

    if task_type == "factoid":
        base = _unique_models([small, mid])
    elif task_type == "market_live" or complexity == "high":
        base = _unique_models([scout, small, mid, large])
    else:
        base = _unique_models([scout, mid, large, kimi])
    if not base:
        base = [settings.model_groq_backup]
    if researchers > len(base):
        while len(base) < researchers:
            base.append(small if len(base) % 2 == 0 else large)
    return [_groq_route(m) for m in base[: max(1, researchers)]]


def _source_hints(query: str) -> list[str]:
    q = query.lower()
    out: list[str] = []
    for token in q.replace("/", " ").split():
        if token.startswith("http://") or token.startswith("https://"):
            domain = token.split("://", 1)[1].split("/", 1)[0].strip().lower()
            if domain:
                hint = f"site:{domain}"
                if hint not in out:
                    out.append(hint)
    defaults = [x.strip() for x in settings.source_hints_default.split(",") if x.strip()]
    for hint in defaults:
        if hint not in out:
            out.append(hint)
    return out[:6]


def build_swarm_config(
    query: str,
    budget_mode: str = "balanced",
    depth: str = "standard",
    provider_pref: str = "auto",
) -> SwarmConfig:
    q = query.lower()
    tokens = len(q.split())
    interrogative = q.split(" ", 1)[0] if q.split() else ""
    is_factoid = interrogative in {"who", "what", "when", "where", "which", "were", "is", "did"} and tokens < 20

    if budget_mode == "low":
        if is_factoid:
            task_type, complexity, researchers, verifier_passes, min_verified_evidence = "factoid", "low", 1, 1, 1
        else:
            task_type, complexity, researchers, verifier_passes, min_verified_evidence = "mixed", "low", 1, 1, 1
    elif budget_mode == "high":
        if is_factoid:
            task_type, complexity, researchers, verifier_passes, min_verified_evidence = "factoid", "low", 2, 1, 2
        elif tokens > 25:
            task_type, complexity, researchers, verifier_passes, min_verified_evidence = "analytical", "high", 4, 2, 4
        else:
            task_type, complexity, researchers, verifier_passes, min_verified_evidence = "mixed", "high", 3, 2, 3
    else:  # balanced
        if is_factoid:
            task_type, complexity, researchers, verifier_passes, min_verified_evidence = "factoid", "low", 2, 1, 1
        elif tokens > 35:
            task_type, complexity, researchers, verifier_passes, min_verified_evidence = "analytical", "high", 3, 2, 3
        else:
            task_type, complexity, researchers, verifier_passes, min_verified_evidence = "mixed", "medium", 2, 2, 2

    # depth="quick" trims resources; depth="deep" adds them.
    if depth == "quick":
        researchers = max(1, researchers - 1)
        verifier_passes = 1
        min_verified_evidence = max(1, min_verified_evidence - 1)
    elif depth == "deep":
        researchers = min(researchers + 2, 6)
        verifier_passes = min(verifier_passes + 1, 3)
        min_verified_evidence = min_verified_evidence + 1

    groq_ready = settings.groq_api_key not in {"", "stub"}
    anthropic_ready = settings.anthropic_api_key not in {"", "stub"}
    together_ready = settings.deepseek_api_key not in {"", "stub"}

    if groq_ready:
        cat = _groq_model_catalog()
        small = cat["reason_small"]
        mid = cat["reason_mid"]
        large = cat["reason_large"]
        kimi = cat["tool_kimi"]

        if budget_mode == "low":
            # Cheapest Groq models everywhere.
            planner = _groq_route(small)
            verifier = _groq_route(small)
            analyst = _groq_route(small)
            writer = _groq_route(mid or small)
            researcher_routes = [_groq_route(small)] * researchers
        elif budget_mode == "high" and anthropic_ready:
            # Opus for planning/analysis, Sonnet for writing, Groq scouts for research.
            planner = {"provider": "anthropic", "model": settings.model_anthropic_opus, "provider_locked": True}
            analyst = {"provider": "anthropic", "model": settings.model_anthropic_opus, "provider_locked": True}
            writer = {"provider": "anthropic", "model": settings.model_anthropic, "provider_locked": True}
            verifier = _groq_route(large)
            researcher_routes = _build_groq_researcher_routes(task_type, complexity, researchers)
        else:
            # Balanced: large Groq for planner/analyst/verifier, mixed for research.
            planner = _groq_route(large if complexity == "high" else mid)
            analyst = _groq_route(large if complexity == "high" else mid)
            writer = _groq_route(kimi or mid)
            verifier = _groq_route(large)
            researcher_routes = _build_groq_researcher_routes(task_type, complexity, researchers)

        researcher = researcher_routes[0]
    elif anthropic_ready:
        # Anthropic-only fallback: use Haiku for bulk work, Sonnet for writing
        haiku = {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "provider_locked": False}
        sonnet = {"provider": "anthropic", "model": settings.model_anthropic, "provider_locked": False}
        opus = {"provider": "anthropic", "model": settings.model_anthropic_opus, "provider_locked": False}
        if budget_mode == "low":
            planner = haiku
            verifier = haiku
            analyst = haiku
            writer = haiku
            researcher_routes = [haiku] * researchers
        elif budget_mode == "high":
            planner = opus
            verifier = sonnet
            analyst = opus
            writer = sonnet
            researcher_routes = [haiku, sonnet][:researchers]
        else:
            planner = sonnet
            verifier = haiku
            analyst = sonnet
            writer = sonnet
            researcher_routes = [haiku, sonnet][:researchers]
        researcher = researcher_routes[0]
    elif together_ready:
        planner = {"provider": "together", "model": settings.model_primary}
        verifier = planner
        analyst = planner
        writer = planner
        researcher = planner
        researcher_routes = [researcher]
    else:
        planner = verifier = analyst = writer = {}
        researcher = planner
        researcher_routes = [researcher]

    return SwarmConfig(
        task_type=task_type,
        complexity=complexity,
        researchers=researchers,
        verifier_passes=verifier_passes,
        min_verified_evidence=min_verified_evidence,
        source_hints=_source_hints(query),
        model_routes={
            "planner": planner,
            "researcher": researcher,
            "analyst": analyst,
            "verifier": verifier,
            "writer": writer,
        },
        researcher_routes=researcher_routes,
    )
