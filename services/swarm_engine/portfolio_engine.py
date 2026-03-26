from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any

from services.swarm_engine.settings import settings
from services.swarm_engine.swarm_config import SwarmConfig, build_swarm_config


@dataclass
class PortfolioLane:
    lane_id: str
    label: str
    provider: str
    strategy: str
    config: SwarmConfig


@dataclass
class PortfolioPreferences:
    budget_mode: str = "balanced"  # low | balanced | high
    depth: str = "standard"  # quick | standard | deep
    provider_pref: str = "auto"  # auto | mixed | groq | together | gemini | openai | anthropic
    lane_preference: str = "auto"  # auto | both | fast | deep
    detail_level: str = "brief"  # brief | detail
    preference_text: str = ""


def _premium_intent(prefs: PortfolioPreferences) -> bool:
    return prefs.budget_mode == "high"


def _csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def provider_order(prefs: PortfolioPreferences) -> list[str]:
    # Configurable orchestration policy via env-backed settings.
    if prefs.provider_pref == "groq":
        return ["groq"]
    if prefs.provider_pref == "together":
        return ["together"]
    if prefs.provider_pref == "gemini":
        return ["gemini"]
    if prefs.provider_pref == "openai":
        return ["openai"]
    if prefs.provider_pref == "anthropic":
        return ["anthropic"]
    if prefs.provider_pref == "mixed":
        if _premium_intent(prefs):
            return _csv_list(settings.provider_order_mixed_high) or ["anthropic", "openai", "groq", "together", "gemini"]
        return _csv_list(settings.provider_order_mixed_balanced) or ["groq", "anthropic", "openai", "together", "gemini"]
    # auto
    if _premium_intent(prefs):
        return _csv_list(settings.provider_order_auto_high) or ["anthropic", "openai", "groq", "together", "gemini"]
    if prefs.budget_mode == "low":
        return _csv_list(settings.provider_order_auto_low) or ["groq", "gemini", "together", "openai", "anthropic"]
    return _csv_list(settings.provider_order_auto_balanced) or ["groq", "anthropic", "openai", "together", "gemini"]


def _provider_ready(provider: str) -> bool:
    if provider == "groq":
        return settings.groq_api_key not in {"", "stub"}
    if provider == "together":
        return settings.deepseek_api_key not in {"", "stub"}
    if provider == "gemini":
        return settings.gemini_api_key not in {"", "stub"}
    if provider == "openai":
        return settings.openai_api_key not in {"", "stub"}
    if provider == "anthropic":
        return settings.anthropic_api_key not in {"", "stub"}
    return False


def _route(provider: str, model: str, locked: bool = True) -> dict[str, Any]:
    return {"provider": provider, "model": model, "provider_locked": locked}


def _clone_cfg(base: SwarmConfig, **kw: Any) -> SwarmConfig:
    return SwarmConfig(
        task_type=kw.get("task_type", base.task_type),
        complexity=kw.get("complexity", base.complexity),
        researchers=int(kw.get("researchers", base.researchers)),
        verifier_passes=int(kw.get("verifier_passes", base.verifier_passes)),
        min_verified_evidence=int(kw.get("min_verified_evidence", base.min_verified_evidence)),
        source_hints=list(kw.get("source_hints", base.source_hints)),
        model_routes=dict(kw.get("model_routes", base.model_routes)),
        researcher_routes=list(kw.get("researcher_routes", base.researcher_routes)),
    )


def build_portfolio_lanes(query: str, prefs: PortfolioPreferences) -> list[PortfolioLane]:
    base = build_swarm_config(query)
    lanes: list[PortfolioLane] = []
    order = provider_order(prefs)

    allow_mixed = prefs.provider_pref in {"auto", "mixed"}
    allow_groq = prefs.provider_pref in {"auto", "mixed", "groq"}
    allow_together = prefs.provider_pref in {"auto", "mixed", "together"}
    allow_gemini = prefs.provider_pref in {"auto", "mixed", "gemini"}
    allow_openai = prefs.provider_pref in {"auto", "mixed", "openai"}
    allow_anthropic = prefs.provider_pref in {"auto", "mixed", "anthropic"}

    if allow_groq and _provider_ready("groq") and "groq" in order:
        lane_pref = (prefs.lane_preference or "auto").strip().lower()
        include_fast = lane_pref in {"auto", "both", "fast"}
        include_deep = lane_pref in {"auto", "both", "deep"}
        # Fast lane: lower-cost models + broad recency scan.
        if include_fast:
            fast_cfg = _clone_cfg(
                base,
                researchers=max(1, base.researchers - 1),
                verifier_passes=1,
                source_hints=base.source_hints + _csv_list(settings.lane_hints_groq_fast),
                model_routes={
                    "planner": _route("groq", settings.model_groq_backup),
                    "researcher": _route("groq", settings.model_groq_backup),
                    "analyst": _route("groq", settings.model_groq_qwen32b or settings.model_groq_backup),
                    "verifier": _route("groq", settings.model_groq_backup),
                    "writer": _route("groq", settings.model_groq_backup),
                },
                researcher_routes=[
                    _route("groq", settings.model_groq_llama_scout or settings.model_groq_backup),
                    _route("groq", settings.model_groq_backup),
                ],
            )
            lanes.append(
                PortfolioLane(
                    lane_id="groq_fast",
                    label="Groq Fast Scan",
                    provider="groq",
                    strategy="broad-recent",
                    config=fast_cfg,
                )
            )

        # Deep lane: higher verification + deeper reasoning.
        if include_deep:
            deep_cfg = _clone_cfg(
                base,
                researchers=max(base.researchers, 3),
                verifier_passes=max(2, base.verifier_passes),
                min_verified_evidence=max(base.min_verified_evidence, 2),
                source_hints=base.source_hints + _csv_list(settings.lane_hints_groq_deep),
                model_routes={
                    "planner": _route("groq", settings.model_groq),
                    "researcher": _route("groq", settings.model_groq_qwen32b or settings.model_groq_backup),
                    "analyst": _route("groq", settings.model_groq),
                    "verifier": _route("groq", settings.model_groq),
                    "writer": _route("groq", settings.model_groq_kimi_k2 or settings.model_groq),
                },
                researcher_routes=[
                    _route("groq", settings.model_groq_llama_scout or settings.model_groq_backup),
                    _route("groq", settings.model_groq_backup),
                    _route("groq", settings.model_groq_qwen32b or settings.model_groq_backup),
                    _route("groq", settings.model_groq),
                ],
            )
            lanes.append(
                PortfolioLane(
                    lane_id="groq_deep",
                    label="Groq Deep Dive",
                    provider="groq",
                    strategy="high-faithfulness",
                    config=deep_cfg,
                )
            )

    if allow_together and _provider_ready("together") and "together" in order:
        t_cfg = _clone_cfg(
            base,
            researchers=max(2, base.researchers),
            source_hints=base.source_hints + _csv_list(settings.lane_hints_together),
            model_routes={
                "planner": _route("together", settings.model_primary),
                "researcher": _route("together", settings.model_primary),
                "analyst": _route("together", settings.model_primary),
                "verifier": _route("together", settings.model_primary),
                "writer": _route("together", settings.model_primary),
            },
            researcher_routes=[_route("together", settings.model_primary)],
        )
        lanes.append(
            PortfolioLane(
                lane_id="together_reasoner",
                label="Together DeepSeek",
                provider="together",
                strategy="reasoning-heavy",
                config=t_cfg,
            )
        )

    if allow_gemini and _provider_ready("gemini") and "gemini" in order:
        g_cfg = _clone_cfg(
            base,
            researchers=max(1, base.researchers - 1),
            verifier_passes=1,
            source_hints=base.source_hints + _csv_list(settings.lane_hints_gemini),
            model_routes={
                "planner": _route("gemini", settings.model_fallback),
                "researcher": _route("gemini", settings.model_fallback),
                "analyst": _route("gemini", settings.model_fallback),
                "verifier": _route("gemini", settings.model_fallback),
                "writer": _route("gemini", settings.model_fallback),
            },
            researcher_routes=[_route("gemini", settings.model_fallback)],
        )
        lanes.append(
            PortfolioLane(
                lane_id="gemini_fast",
                label="Gemini Fast",
                provider="gemini",
                strategy="recency-oriented",
                config=g_cfg,
            )
        )

    if allow_openai and _provider_ready("openai") and "openai" in order:
        o_cfg = _clone_cfg(
            base,
            researchers=max(2, base.researchers),
            verifier_passes=max(2, base.verifier_passes),
            source_hints=base.source_hints + _csv_list(settings.lane_hints_openai),
            model_routes={
                "planner": _route("openai", settings.model_openai),
                "researcher": _route("openai", settings.model_openai),
                "analyst": _route("openai", settings.model_openai),
                "verifier": _route("openai", settings.model_openai),
                "writer": _route("openai", settings.model_openai),
            },
            researcher_routes=[_route("openai", settings.model_openai)],
        )
        lanes.append(
            PortfolioLane(
                lane_id="openai_reasoner",
                label="OpenAI Reasoner",
                provider="openai",
                strategy="balanced-reasoning",
                config=o_cfg,
            )
        )

    if allow_anthropic and _provider_ready("anthropic") and "anthropic" in order:
        a_cfg = _clone_cfg(
            base,
            researchers=max(2, base.researchers),
            verifier_passes=max(2, base.verifier_passes),
            source_hints=base.source_hints + _csv_list(settings.lane_hints_anthropic),
            model_routes={
                "planner": _route("anthropic", settings.model_anthropic),
                "researcher": _route("anthropic", settings.model_anthropic),
                "analyst": _route("anthropic", settings.model_anthropic),
                "verifier": _route("anthropic", settings.model_anthropic),
                "writer": _route("anthropic", settings.model_anthropic),
            },
            researcher_routes=[_route("anthropic", settings.model_anthropic)],
        )
        lanes.append(
            PortfolioLane(
                lane_id="anthropic_reasoner",
                label="Claude Reasoner",
                provider="anthropic",
                strategy="deep-interpretive",
                config=a_cfg,
            )
        )

    if not lanes:
        lanes = [PortfolioLane(lane_id="default", label="Default", provider="auto", strategy="default", config=base)]
    else:
        # Keep only providers from preferred order and sort accordingly.
        rank = {p: i for i, p in enumerate(order)}
        lanes = sorted(lanes, key=lambda l: rank.get(l.provider, 999))

    # Budget/depth pruning.
    if prefs.budget_mode == "low":
        lanes = lanes[:2]
    elif prefs.budget_mode == "balanced":
        lanes = lanes[:3]
    elif prefs.budget_mode == "high" and len(lanes) < 4 and allow_mixed:
        lanes = lanes

    if prefs.depth == "quick":
        # Keep only cheaper/faster lanes.
        lanes = [l for l in lanes if "fast" in l.lane_id or l.provider in {"gemini"}] or lanes[:2]
    elif prefs.depth == "deep" and len(lanes) >= 2:
        lanes = [l for l in lanes if "deep" in l.lane_id or l.provider in {"together", "groq"}] or lanes

    return lanes


def _model_weight(model_name: str) -> float:
    m = model_name.lower()
    if "120b" in m:
        return 1.0
    if "opus" in m:
        return 1.0
    if "kimi" in m or "deepseek" in m:
        return 0.85
    if "sonnet" in m or "claude" in m:
        return 0.8
    if "32b" in m or "70b" in m:
        return 0.6
    if "scout" in m or "gemini" in m:
        return 0.5
    if "20b" in m or "haiku" in m or "8b" in m:
        return 0.35
    return 0.55


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _keyword_coverage(question: str, answer: str) -> float:
    if not answer:
        return 0.0
    words = [w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 3]
    if not words:
        return 0.5
    ans = answer.lower()
    hit = sum(1 for w in set(words) if w in ans)
    return min(1.0, hit / max(1, len(set(words))))


def extract_requirement_facets(question: str) -> list[str]:
    q_raw = question.strip()
    q = q_raw.lower()
    facets: list[str] = []

    def _add(x: str) -> None:
        x = x.strip().lower()
        if not x or x in facets:
            return
        facets.append(x)

    for m in re.findall(r"\b[\w.-]+\.md\b", q):
        _add(m)

    for m in re.findall(r"\br/[a-z0-9_]+\b", q):
        _add(m)

    for m in re.findall(r"\b[A-Z][a-z]{4,}\b", q_raw):
        _add(m.lower())

    for m in re.findall(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b", q_raw):
        _add(m.lower())

    quoted = re.findall(r"\"([^\"]{3,80})\"|'([^']{3,80})'", q_raw)
    for a, b in quoted:
        phrase = (a or b).strip().lower()
        if phrase:
            _add(phrase)

    if re.search(r"\bx\b", q):
        _add("x")

    for token in re.findall(r"[a-z0-9][a-z0-9_.-]{5,}", q):
        if token.endswith((".com", ".org", ".ai", ".io")):
            _add(token)
            continue
        _add(token)

    if len(facets) > 10:
        facets = facets[:10]
    return facets


def missing_requirement_facets(question: str, evidence_items: list[dict[str, Any]]) -> list[str]:
    facets = extract_requirement_facets(question)
    if not facets:
        return []
    ev_blob = " ".join(
        f"{str(i.get('title', ''))} {str(i.get('source_url', ''))} {str(i.get('summary', ''))}".lower()
        for i in (evidence_items or [])
    )
    missing: list[str] = []
    for f in facets:
        if f == "x":
            ok = bool(re.search(r"\bx\b", ev_blob))
        else:
            ok = f in ev_blob
        if not ok:
            missing.append(f)
    return missing


def _requirement_coverage(question: str, state: dict[str, Any]) -> float:
    facets = extract_requirement_facets(question)
    if not facets:
        return 1.0
    md = str(state.get("final_markdown", "")).lower()
    missing = set(missing_requirement_facets(question, state.get("evidence_items", []) or []))
    hit = 0
    for f in facets:
        ok = f not in missing and (bool(re.search(r"\bx\b", md)) if f == "x" else (f in md))
        hit += 1 if ok else 0
    return hit / max(1, len(facets))


def _recency_score(evidence_items: list[dict[str, Any]]) -> float:
    if not evidence_items:
        return 0.0
    now = datetime.now(timezone.utc).timestamp()
    horizon = 180 * 24 * 3600
    recent = 0
    parsed = 0
    for item in evidence_items:
        ts = str(item.get("timestamp", "")).strip()
        try:
            tsv = float(ts)
            parsed += 1
            if now - tsv <= horizon:
                recent += 1
        except Exception:
            continue
    if parsed == 0:
        return 0.4
    return max(0.0, min(1.0, recent / parsed))


def _novelty_score(markdown: str) -> float:
    if not markdown:
        return 0.0
    text = markdown.lower()
    cue = 0
    for k in ["novel", "hypothesis", "scenario", "out-of-the-box", "non-obvious", "counterfactual"]:
        if k in text:
            cue += 1
    uniq = len(set(re.findall(r"[a-z0-9]+", text)))
    total = max(1, len(re.findall(r"[a-z0-9]+", text)))
    diversity = uniq / total
    return max(0.0, min(1.0, 0.55 * min(1.0, cue / 3.0) + 0.45 * min(1.0, diversity * 6.0)))


def _cost_estimate(state: dict[str, Any], lane: PortfolioLane) -> float:
    plan = state.get("plan", [])
    cycle = max(1, int(state.get("cycle_count", 1)))
    token_plan = 0.0
    for p in plan:
        token_plan += _safe_float(p.get("expected_cost_tokens", 1200), 1200)
    main_model = str(lane.config.model_routes.get("writer", {}).get("model", ""))
    model_mult = _model_weight(main_model)
    return token_plan * cycle * model_mult


def compute_candidate_metrics(
    question: str,
    lane: PortfolioLane,
    state: dict[str, Any],
    latency_s: float,
) -> dict[str, float]:
    metrics = state.get("metrics", {}) or {}
    faithfulness = _safe_float(metrics.get("rag_faithfulness", 0.0))
    req_cov = _requirement_coverage(question, state)
    lexical_cov = _keyword_coverage(question, state.get("final_markdown", ""))
    coverage = 0.35 * _safe_float(metrics.get("plan_adherence", 0.0)) + 0.25 * lexical_cov + 0.40 * req_cov
    recency = _recency_score(state.get("evidence_items", []))
    novelty = _novelty_score(state.get("final_markdown", ""))
    cost = _cost_estimate(state, lane)
    latency = max(0.001, latency_s)
    return {
        "faithfulness": round(max(0.0, min(1.0, faithfulness)), 4),
        "coverage": round(max(0.0, min(1.0, coverage)), 4),
        "recency": round(max(0.0, min(1.0, recency)), 4),
        "novelty": round(max(0.0, min(1.0, novelty)), 4),
        "cost": round(cost, 2),
        "latency": round(latency, 2),
    }


def pareto_frontier(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []

    # maximize first 4 metrics; minimize cost/latency
    def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
        am = a["score_vector"]
        bm = b["score_vector"]
        ge = (
            am["faithfulness"] >= bm["faithfulness"]
            and am["coverage"] >= bm["coverage"]
            and am["recency"] >= bm["recency"]
            and am["novelty"] >= bm["novelty"]
            and am["cost"] <= bm["cost"]
            and am["latency"] <= bm["latency"]
        )
        strict = (
            am["faithfulness"] > bm["faithfulness"]
            or am["coverage"] > bm["coverage"]
            or am["recency"] > bm["recency"]
            or am["novelty"] > bm["novelty"]
            or am["cost"] < bm["cost"]
            or am["latency"] < bm["latency"]
        )
        return ge and strict

    out = []
    for c in candidates:
        if any(dominates(o, c) for o in candidates if o is not c):
            continue
        out.append(c)
    return out


def choose_candidate(
    frontier: list[dict[str, Any]],
    prefs: PortfolioPreferences,
) -> dict[str, Any] | None:
    if not frontier:
        return None

    def utility(c: dict[str, Any]) -> float:
        s = c["score_vector"]
        if prefs.budget_mode == "low":
            return 0.30 * s["faithfulness"] + 0.20 * s["coverage"] + 0.10 * s["novelty"] - 0.30 * (s["cost"] / 5000.0) - 0.10 * (s["latency"] / 120.0)
        if prefs.depth == "deep":
            return 0.35 * s["faithfulness"] + 0.30 * s["coverage"] + 0.15 * s["recency"] + 0.20 * s["novelty"] - 0.05 * (s["cost"] / 5000.0)
        return 0.30 * s["faithfulness"] + 0.25 * s["coverage"] + 0.15 * s["recency"] + 0.15 * s["novelty"] - 0.10 * (s["cost"] / 5000.0) - 0.05 * (s["latency"] / 120.0)

    scored = sorted(frontier, key=utility, reverse=True)
    return scored[0]


def qualitative_band(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    if score > 0:
        return "low"
    return "none"
