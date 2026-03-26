from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.checkpoint.sqlite import SqliteSaver
except Exception:  # pragma: no cover
    InMemorySaver = object
    SqliteSaver = None
    START = "__start__"
    END = "__end__"

    class _SimpleCompiledGraph:
        def __init__(self, nodes, edges, cond_edges):
            self.nodes = nodes
            self.edges = edges
            self.cond_edges = cond_edges

        async def ainvoke(self, state, config=None):
            current = self.edges.get(START)
            cur_state = state
            while current and current != END:
                cur_state = await self.nodes[current](cur_state)
                if current in self.cond_edges:
                    fn, mapping = self.cond_edges[current]
                    current = mapping[fn(cur_state)]
                else:
                    current = self.edges.get(current, END)
            return cur_state

    class StateGraph:  # type: ignore[misc]
        def __init__(self, _state_type):
            self.nodes = {}
            self.edges = {}
            self.cond_edges = {}

        def add_node(self, name, fn):
            self.nodes[name] = fn

        def add_edge(self, src, dst):
            self.edges[src] = dst

        def add_conditional_edges(self, src, fn, mapping):
            self.cond_edges[src] = (fn, mapping)

        def compile(self, checkpointer=None):
            return _SimpleCompiledGraph(self.nodes, self.edges, self.cond_edges)

from services.swarm_engine.llm import LLMAdapter
from services.logging import get_logger
from services.canvas.interface import generate_knowledge_map
from services.swarm_engine.artifact_exports import maybe_export_artifacts
from services.memory.umem_adapter import mem_bootstrap_pack, mem_save_note, mem_search_pack
from services.swarm_engine.portfolio_engine import (
    PortfolioPreferences,
    build_portfolio_lanes,
    choose_candidate,
    compute_candidate_metrics,
    missing_requirement_facets,
    pareto_frontier,
    provider_order,
    qualitative_band,
)
from services.swarm_engine.embeddings import EmbeddingAdapter
from services.swarm_engine.settings import llm_mock_enabled, settings
from services.swarm_engine.swarm_config import SwarmConfig, build_swarm_config
from services.swarm_engine.state import SwarmState
from services.swarm_engine.tracing import emit_trace, langsmith_env
from services.supabase_store import supabase_store
from tools.mcp_tools import (
    ask_user_question,
    open_url,
    playwright_fetch,
    qdrant_search,
    qdrant_upsert,
    snapshot_url,
    summarize_text,
    tool_registry,
    wikipedia_lookup,
    web_search,
)


class SessionRuntime:
    def __init__(self) -> None:
        self.queues: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self.status: dict[str, str] = {}
        self.final_state: dict[str, SwarmState] = {}
        self.hitl_questions: dict[str, str] = {}
        self.hitl_answers: dict[str, str] = {}
        self.steer_messages: dict[str, list[str]] = defaultdict(list)
        self.portfolio_children: dict[str, list[str]] = defaultdict(list)

    async def emit(self, session_id: str, event: dict[str, Any]) -> None:
        await self.queues[session_id].put(event)


runtime = SessionRuntime()
logger = get_logger("swarm.runtime")
llm = LLMAdapter(
    use_mock=llm_mock_enabled(),
    deepseek_api_key=settings.deepseek_api_key,
    gemini_api_key=settings.gemini_api_key,
    groq_api_key=settings.groq_api_key,
    openai_api_key=settings.openai_api_key,
    anthropic_api_key=settings.anthropic_api_key,
    model_primary=settings.model_primary,
    model_fallback=settings.model_fallback,
    model_groq=settings.model_groq,
    model_groq_backup=settings.model_groq_backup,
    model_groq_qwen32b=settings.model_groq_qwen32b,
    model_groq_llama_scout=settings.model_groq_llama_scout,
    model_groq_kimi_k2=settings.model_groq_kimi_k2,
    model_openai=settings.model_openai,
    model_anthropic=settings.model_anthropic,
    model_anthropic_opus=settings.model_anthropic_opus,
)
embedder = EmbeddingAdapter(
    api_key=settings.deepseek_api_key,
    model=settings.embedding_model,
    use_mock=llm.use_mock,
)


_DEFER_PATTERNS = [p.strip() for p in settings.hitl_defer_patterns.split(",") if p.strip()]
_HITL_APPROVAL_KEYWORDS = {k.strip().lower() for k in settings.hitl_approval_keywords.split(",") if k.strip()}


def _available_models_summary(budget_mode: str = "balanced") -> str:
    """Build a budget-aware model catalog for the planner prompt."""
    budget_guidance = {
        "low": (
            "Budget: LOW — optimize for cost. Use the cheapest/fastest models everywhere. "
            "1-2 researchers, 1 verifier pass, small models for all roles."
        ),
        "balanced": (
            "Budget: BALANCED — balance quality and cost. Use mid-size models for planning/analysis, "
            "fast scouts for research. 2-3 researchers."
        ),
        "high": (
            "Budget: HIGH — maximize quality. Use the most capable reasoning models for planner/analyst. "
            "Use synthesis-specialized models for writer. Diverse researcher lanes. "
            "2 verifier passes. Cost is not a concern."
        ),
    }.get(budget_mode, "Budget: BALANCED")

    lines = [budget_guidance, "", "Available providers and models (ONLY use these in agent_config):"]

    if settings.groq_api_key not in {"", "stub"}:
        groq_models = [
            (settings.model_groq,           "large — best Groq reasoning [balanced+/analyst/planner]"),
            (settings.model_groq_backup,     "small/fast — cheapest [low-budget/researcher]"),
            (settings.model_groq_qwen32b,    "mid-size versatile [balanced researcher]"),
            (settings.model_groq_llama_scout,"scout — fastest retrieval [researcher lanes]"),
            (settings.model_groq_kimi_k2,    "strong reasoning + synthesis [balanced writer/analyst]"),
        ]
        model_lines = [f"    {m} — {desc}" for m, desc in groq_models if m]
        lines.append("  groq:\n" + "\n".join(model_lines))

    if settings.anthropic_api_key not in {"", "stub"}:
        groq_available = settings.groq_api_key not in {"", "stub"}
        if budget_mode == "high" or not groq_available:
            lines.append(
                f"  anthropic:\n"
                f"    {settings.model_anthropic_opus} — Opus, best reasoning [planner/analyst]\n"
                f"    {settings.model_anthropic} — Sonnet, best synthesis [writer]\n"
                f"    claude-haiku-4-5-20251001 — Haiku, fast/cheap [researcher/verifier]"
            )
        else:
            lines.append(
                f"  anthropic (expensive — prefer groq when available):\n"
                f"    {settings.model_anthropic_opus} — Opus [planner/analyst, high-budget only]\n"
                f"    {settings.model_anthropic} — Sonnet [writer, high-budget only]"
            )

    if settings.openai_api_key not in {"", "stub"}:
        lines.append(f"  openai:\n    {settings.model_openai} — balanced [writer/analyst]")
    if settings.deepseek_api_key not in {"", "stub"}:
        lines.append(f"  together:\n    {settings.model_primary} — deep reasoning")
    if settings.gemini_api_key not in {"", "stub"}:
        lines.append(f"  gemini:\n    {settings.model_fallback} — large context")

    role_guidance = {
        "low": [
            "  planner   → cheapest small model",
            "  researcher → single fast scout lane",
            "  verifier  → small model, 1 pass",
            "  analyst   → small model",
            "  writer    → mid model",
        ],
        "balanced": [
            "  planner   → large Groq or mid-size (gpt-oss-120b, kimi-k2)",
            "  researcher → 2-3 lanes: scout + mid, DIFFERENT models for diversity",
            "  verifier  → large model, 1-2 passes",
            "  analyst   → large Groq (gpt-oss-120b, kimi-k2)",
            "  writer    → kimi-k2 or gpt-oss-120b",
        ],
        "high": [
            "  planner   → claude-opus-4-6 (best reasoning)",
            "  researcher → 3-4 diverse lanes: mix groq scout + mid + kimi",
            "  verifier  → gpt-oss-120b or claude-sonnet, 2 passes",
            "  analyst   → claude-opus-4-6",
            "  writer    → claude-sonnet-4-6 (best synthesis)",
        ],
    }.get(budget_mode, [])

    lines += ["", "Role routing guidance (follow budget tier):"] + role_guidance
    return "\n".join(lines)


def _step_budget_for_mode(budget_mode: str) -> int:
    if budget_mode == "low":
        return max(1, int(settings.max_steps_low))
    if budget_mode == "high":
        return max(1, int(settings.max_steps_high))
    return max(1, int(settings.max_steps_balanced))


def _is_defer_answer(answer: str) -> bool:
    text = answer.strip().lower()
    if not text:
        return False
    return any(re.search(pat, text) for pat in _DEFER_PATTERNS)


def _should_ask_hitl(state: SwarmState, stage: str, question: str) -> bool:
    if state.get("autonomous_mode", False):
        return False
    skipped = set(state.get("hitl_opt_out_stages", []))
    if stage in skipped:
        return False
    last_stage = state.get("hitl_last_stage", "")
    last_question = state.get("hitl_last_question", "")
    if last_stage == stage and last_question and last_question.strip().lower() == question.strip().lower():
        return False
    return True


def _event(session_id: str, node_name: str, payload: dict[str, Any], event_type: str) -> dict[str, Any]:
    trace = emit_trace(session_id=session_id, node_name=node_name, payload=payload)
    return {
        "type": event_type,
        "trace_id": trace["trace_id"],
        "node_name": node_name,
        "timestamp": trace["timestamp"],
        "payload": payload,
        "cost_estimate": payload.get("cost_estimate", 0),
    }


async def _agent_message(session_id: str, sender: str, receiver: str, message: str) -> None:
    await runtime.emit(
        session_id,
        _event(
            session_id,
            "AgentHub",
            {"from": sender, "to": receiver, "message": message[:400]},
            "agent_message",
        ),
    )


def _render_orchestration_tree(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""
    row = history[-1]
    step = int(row.get("step", len(history)))
    action = str(row.get("next_action", ""))
    reason = str(row.get("rationale", "")).strip()
    low_reason = reason.lower()
    tag = ""
    if any(k in low_reason for k in ["no result", "not found", "insufficient", "gap", "pivot"]):
        tag = " [pivot]"
    if action == "end" and any(k in low_reason for k in ["no evidence", "unresolved", "budget exhausted"]):
        tag = " [dead-end]"
    return f"└─ [{step:02d}] {action}{tag}" + (f"  ({reason[:80]})" if reason else "")


_META_QUERY_PATTERNS = [
    r"\bself[-\s]?conscious",
    r"\bself[-\s]?reflection",
    r"\binternal (?:reasoning|monologue|thought)",
    r"\bas an ai\b",
    r"\bi know the user\b",
    r"\bpersona\b",
]


def _clean_query_text(text: str) -> str:
    return " ".join(str(text or "").replace("\n", " ").replace("\t", " ").split()).strip()


def _is_meta_query_text(text: str) -> bool:
    t = _clean_query_text(text).lower()
    if not t:
        return True
    return any(re.search(p, t) for p in _META_QUERY_PATTERNS)


def _build_subtask_query(user_query: str, subtask: dict[str, Any]) -> str:
    subqs = subtask.get("subquestions", [])
    candidates: list[str] = []
    if isinstance(subqs, list):
        candidates.extend(str(x) for x in subqs if str(x).strip())
    desc = str(subtask.get("description", "")).strip()
    if desc:
        candidates.append(desc)
        candidates.append(f"{user_query} {desc}")
    candidates.append(user_query)
    for c in candidates:
        cleaned = _clean_query_text(c)
        if not cleaned:
            continue
        if _is_meta_query_text(cleaned):
            continue
        return cleaned
    return _clean_query_text(user_query)


async def orchestrator_node(state: SwarmState) -> SwarmState:
    budget_left = int(state.get("budget_steps_remaining", 0))
    steps_used = int(state.get("budget_steps_used", 0)) + 1
    budget_left = max(0, budget_left - 1)
    budget_exhausted = budget_left <= 0
    route = state.get("model_routes", {}).get("planner", {})

    items = state.get("evidence_items", [])
    verified = [i for i in items if i.get("verification_score", 0) > 0]
    avg_score = (sum(i["verification_score"] for i in verified) / len(verified)) if verified else 0.0
    snapshot = {
        "has_plan": bool(state.get("plan")),
        "plan_items": len(state.get("plan", [])),
        "plan_done": sum(1 for p in state.get("plan", []) if p.get("status") == "done"),
        "current_plan_index": int(state.get("current_plan_index", 0)),
        "evidence_count": len(items),
        "verified_count": len(verified),
        "avg_verification_score": round(avg_score, 2),
        "contradictions": sum(1 for i in verified if i.get("contradiction")),
        "has_verification": bool(state.get("verification")),
        "has_final_markdown": bool(state.get("final_markdown")),
        "needs_more_research": bool(state.get("needs_more_research", False)),
        "needs_revision": bool(state.get("needs_revision", False)),
        "last_worker": state.get("last_worker", ""),
        "cycle_count": int(state.get("cycle_count", 0)),
        "budget_steps_remaining": budget_left,
        "budget_exhausted": budget_exhausted,
        "postprocessed": bool(state.get("postprocessed", False)),
        "save_done": bool(state.get("save_done", False)),
        "has_metrics": bool(state.get("metrics")),
        "researcher_exhausted": bool(state.get("researcher_exhausted", False)),
    }
    decision = await llm.orchestrate(
        query=state.get("user_query", ""),
        state_snapshot=snapshot,
        allowed_actions=["planner", "researcher", "verifier", "analyst", "writer", "hitl_review", "postprocess", "save", "metrics", "end"],
        route=route,
    )
    next_action = str(decision.get("next_action", "researcher"))
    rationale = str(decision.get("rationale", "")).strip()
    focus_query = str(decision.get("focus_query", "")).strip()
    replan = next_action == "planner" and bool(state.get("plan"))
    req_batch = decision.get("research_batch_size")
    if isinstance(req_batch, int) and req_batch > 0:
        state = {**state, "research_batch_size": req_batch}
    verify_indices = decision.get("verify_indices")
    if isinstance(verify_indices, list):
        state = {**state, "verifier_focus_indices": verify_indices}
    post_final_iters = int(state.get("post_final_iterations", 0))

    # Safety rails: only enforce graph/flow correctness.
    if state.get("pending_hitl"):
        next_action = "end_waiting"
        rationale = rationale or "awaiting human input"
    elif not state.get("plan"):
        next_action = "planner"
        rationale = rationale or "need plan before execution"
    elif next_action == "save" and not state.get("postprocessed", False):
        next_action = "postprocess"
        rationale = rationale or "save requires postprocess"
    elif next_action == "metrics" and not state.get("save_done", False):
        next_action = "save" if state.get("postprocessed", False) else "postprocess"
        rationale = rationale or "metrics requires persisted report"
    elif next_action == "end" and not state.get("metrics"):
        next_action = "metrics" if state.get("save_done", False) else ("save" if state.get("postprocessed", False) else "postprocess")
        rationale = rationale or "end requires final metrics"
    elif budget_exhausted and next_action in {"planner", "researcher", "verifier", "analyst"} and not state.get("final_markdown"):
        next_action = "writer" if items else "end"
        rationale = rationale or "budget exhausted, forcing synthesis"
    elif state.get("researcher_exhausted", False) and next_action == "researcher":
        if state.get("evidence_items"):
            next_action = "writer"
            rationale = rationale or "research exhausted with evidence; synthesize"
        elif int(state.get("replan_attempts", 0)) < 1:
            next_action = "planner"
            replan = True
            state = {**state, "replan_attempts": int(state.get("replan_attempts", 0)) + 1}
            rationale = rationale or "research exhausted; trigger one replan"
        else:
            next_action = "writer"
            rationale = rationale or "research exhausted after replan; produce failure-aware writeup"

    # Finalization rails: once a draft exists, progress strictly forward.
    if state.get("final_markdown"):
        if state.get("postprocessed", False) and not state.get("save_done", False):
            next_action = "save"
            rationale = rationale or "postprocess complete; persist report"
        elif state.get("save_done", False) and not state.get("metrics"):
            next_action = "metrics"
            rationale = rationale or "report saved; emit metrics"
        elif state.get("save_done", False) and state.get("metrics"):
            next_action = "end"
            rationale = rationale or "report finalized"

    # Prevent endless post-write loops while still allowing a small re-research window.
    if state.get("final_markdown"):
        if next_action == "writer" and not state.get("needs_revision", False):
            next_action = "postprocess"
            rationale = rationale or "draft already exists; moving to finalization"
            post_final_iters = 0
        elif next_action in {"researcher", "verifier", "analyst", "planner", "writer", "postprocess"}:
            post_final_iters += 1
            if budget_exhausted or post_final_iters > 2:
                next_action = "postprocess"
                rationale = rationale or "closing post-write loop"
        else:
            post_final_iters = 0

    if focus_query:
        state = {**state, "focus_query": focus_query}

    await runtime.emit(
        state["session_id"],
        _event(
            state["session_id"],
            "Orchestrator",
            {
                "next_action": next_action,
                "budget_steps_remaining": budget_left,
                "budget_steps_used": steps_used,
                "budget_exhausted": budget_exhausted,
                "rationale": rationale[:260],
                "last_worker": state.get("last_worker", ""),
                "state_metrics": f"E:{len(items)} V:{len(verified)}",
            },
            "trace",
        ),
    )
    await runtime.emit(
        state["session_id"],
        _event(
            state["session_id"],
            "Orchestrator",
            {
                "summary": f"Orchestrator routing to {next_action}",
                "main_reason": rationale[:180] if rationale else "LLM-driven routing",
            },
            "thought_stream",
        ),
    )
    history = list(state.get("orchestration_history", []))
    history.append(
        {
            "step": steps_used,
            "next_action": next_action,
            "budget_left": budget_left,
            "rationale": rationale[:200],
        }
    )
    tree_text = _render_orchestration_tree(history)
    if tree_text:
        await runtime.emit(
            state["session_id"],
            _event(
                state["session_id"],
                "Orchestrator",
                {"tree": tree_text},
                "orchestration_tree",
            ),
        )
    return {
        **state,
        "orchestrator_next": next_action,
        "budget_steps_remaining": budget_left,
        "budget_steps_used": steps_used,
        "budget_exhausted": budget_exhausted,
        "orchestration_history": history[-100:],
        "force_replan": replan,
        "post_final_iterations": post_final_iters,
        "researcher_exhausted": False if next_action in {"planner", "researcher"} else bool(state.get("researcher_exhausted", False)),
    }


async def planner_node(state: SwarmState) -> SwarmState:
    if state.get("plan_approved") and state.get("plan") and not state.get("force_replan", False):
        return {**state, "pending_hitl": False}

    route = state.get("model_routes", {}).get("planner", {})
    planner_context = state.get("memory_context", "")
    search_res = web_search(
        query=state["user_query"],
        num=max(1, int(settings.planner_search_results)),
        searchapi_api_key=settings.searchapi_api_key,
        tavily_api_key=settings.tavily_api_key,
        use_mock=llm.use_mock,
    )
    if search_res.get("status") in {"ok", "degraded"}:
        snippets = []
        max_snippets = max(1, int(settings.planner_context_snippets))
        for row in search_res.get("payload", {}).get("results", [])[:max_snippets]:
            snippets.append(f"- {row.get('title','')}: {row.get('snippet','')}")
        planner_context = "\n".join(snippets)
    plan, agent_config = await llm.plan(
        state["user_query"],
        route=route,
        context=planner_context,
        available_models=_available_models_summary(state.get("budget_mode", "balanced")),
        available_tools="\n".join(
            f"- {t.get('name', '')}: {t.get('description', '')}".strip()
            for t in state.get("tool_registry", [])
            if t.get("name")
        ),
    )
    max_items = max(1, int(settings.max_plan_items))
    normalized_plan: list[dict[str, Any]] = []
    for i, step in enumerate(plan):
        if isinstance(step, dict):
            normalized_plan.append(
                {
                    "id": str(step.get("id", f"P{i+1}")),
                    "description": str(step.get("description", step.get("title", f"Step {i+1}"))),
                    "subquestions": step.get("subquestions") if isinstance(step.get("subquestions"), list) else [],
                    "tools": step.get("tools") if isinstance(step.get("tools"), list) else [],
                    "expected_cost_tokens": int(step.get("expected_cost_tokens", 200)),
                    "priority": int(step.get("priority", i + 1)),
                    "status": str(step.get("status", "pending")),
                }
            )
        else:
            normalized_plan.append(
                {
                    "id": f"P{i+1}",
                    "description": str(step),
                    "subquestions": [],
                    "tools": [],
                    "expected_cost_tokens": 200,
                    "priority": i + 1,
                    "status": "pending",
                }
            )
    plan = normalized_plan
    if len(plan) > max_items:
        plan = plan[:max_items]

    # Apply planner-decided routing, falling back to swarm_config defaults when absent.
    new_model_routes = dict(state.get("model_routes", {}))
    if agent_config.get("model_routes"):
        new_model_routes.update(agent_config["model_routes"])

    new_researcher_routes = list(state.get("researcher_routes", []))
    if agent_config.get("researcher_routes"):
        new_researcher_routes = agent_config["researcher_routes"]

    new_verifier_passes = agent_config.get("verifier_passes") or state.get("verifier_passes", 1)
    new_min_verified = agent_config.get("min_verified_evidence") or state.get("min_verified_evidence", 2)
    new_researchers = agent_config.get("researchers") or state.get("researchers", 2)
    new_search_results_per_query = agent_config.get("search_results_per_query") or state.get("search_results_per_query", 10)
    new_url_fetch_limit = agent_config.get("url_fetch_limit") or state.get("url_fetch_limit", 20)
    new_docs_per_subtask = agent_config.get("docs_per_subtask") or state.get("docs_per_subtask", 8)
    routing_rationale = agent_config.get("rationale", "")

    event = _event(
        state["session_id"],
        "Planner",
        {
            "plan": plan,
            "task_type": state.get("task_type"),
            "complexity": state.get("complexity"),
            "research_batch_size": state.get("research_batch_size"),
            "model_routes": new_model_routes,
            "researcher_routes": new_researcher_routes,
            "agent_config": agent_config,
        },
        "plan_update",
    )
    await runtime.emit(state["session_id"], event)
    await runtime.emit(
        state["session_id"],
        _event(
            state["session_id"],
            "Planner",
            {
                "summary": "Planned execution strategy with conservative + exploratory subtasks.",
                "steps": [str(p.get("description", ""))[:140] for p in plan[:4]],
                "knowns": f"task_type={state.get('task_type')} complexity={state.get('complexity')}",
                "routing_rationale": routing_rationale,
                "model_routes": {role: r.get("model", "") for role, r in new_model_routes.items()},
            },
            "reasoning_summary",
        ),
    )
    await _agent_message(state["session_id"], "Planner", "Researcher", "Plan prepared. Execute subtasks with grounding + contradiction checks.")
    should_ask = bool(agent_config.get("ask_hitl", False))
    question = str(agent_config.get("hitl_question", "")).strip()
    if should_ask and question and _should_ask_hitl(state, "plan", question):
        runtime.hitl_questions[state["session_id"]] = question
        ask = ask_user_question(question)
        await runtime.emit(state["session_id"], _event(state["session_id"], "HITL_Approval", ask["payload"], "hitl_request"))
    else:
        should_ask = False
        question = ""
    return {
        **state,
        "plan": plan,
        "model_routes": new_model_routes,
        "researcher_routes": new_researcher_routes,
        "verifier_passes": new_verifier_passes,
        "min_verified_evidence": new_min_verified,
        "research_batch_size": new_researchers,
        "search_results_per_query": new_search_results_per_query,
        "url_fetch_limit": new_url_fetch_limit,
        "docs_per_subtask": new_docs_per_subtask,
        "analyst_min_score": agent_config.get("analyst_min_score") or settings.analyst_good_score,
        "writer_sections": agent_config.get("writer_sections") or [],
        "pending_hitl": should_ask,
        "plan_approved": not should_ask,
        "interrupt_stage": "plan" if should_ask else "",
        "hitl_question": question if should_ask else "",
        "hitl_last_stage": "plan" if should_ask else state.get("hitl_last_stage", ""),
        "hitl_last_question": question if should_ask else state.get("hitl_last_question", ""),
        "cycle_count": 0,
        "current_plan_index": 0,
        "force_replan": False,
        "researcher_exhausted": False,
        "queries_asked": [],
    }


async def hitl_node(state: SwarmState) -> SwarmState:
    session_id = state["session_id"]
    if not state.get("pending_hitl"):
        return {**state, "pending_hitl": False}
    stage = state.get("interrupt_stage", "plan")
    answer = runtime.hitl_answers.pop(session_id, None)
    if answer:
        cleaned = answer.strip()
        skipped = set(state.get("hitl_opt_out_stages", []))
        low = cleaned.lower()
        approved = low in _HITL_APPROVAL_KEYWORDS
        if _is_defer_answer(cleaned):
            skipped.add(stage)
            await runtime.emit(session_id, _event(session_id, "HITL_Approval", {"answer": cleaned, "mode": "autonomous"}, "trace"))
            if stage == "plan":
                return {
                    **state,
                    "pending_hitl": False,
                    "hitl_answer": cleaned,
                    "plan_approved": True,
                    "interrupt_stage": "",
                    "hitl_opt_out_stages": sorted(skipped),
                    "autonomous_mode": True,
                }
            return {
                **state,
                "pending_hitl": False,
                "hitl_answer": cleaned,
                "interrupt_stage": "",
                "hitl_opt_out_stages": sorted(skipped),
                "autonomous_mode": True,
            }
        event = _event(session_id, "HITL_Approval", {"answer": answer}, "trace")
        await runtime.emit(session_id, event)
        if stage == "plan":
            return {**state, "pending_hitl": False, "hitl_answer": answer, "plan_approved": True, "interrupt_stage": ""}
        if approved:
            skipped.add(stage)
            return {
                **state,
                "pending_hitl": False,
                "hitl_answer": cleaned,
                "interrupt_stage": "",
                "hitl_opt_out_stages": sorted(skipped),
            }
        return {
            **state,
            "pending_hitl": False,
            "hitl_answer": cleaned,
            "focus_query": cleaned,
            "interrupt_stage": "",
        }
    return {**state, "pending_hitl": True}


async def researcher_node(state: SwarmState) -> SwarmState:
    evidence = state.get("evidence_items", [])
    start_evidence_count = len(evidence)
    search_failures = list(state.get("search_failures", []))
    plan = state["plan"]
    idx = state.get("current_plan_index", 0)
    batch = max(1, int(state.get("research_batch_size") or 1))
    if idx >= len(plan):
        await runtime.emit(
            state["session_id"],
            _event(
                state["session_id"],
                "Researcher",
                {"researcher_exhausted": True, "reason": "plan_exhausted_no_remaining_subtasks"},
                "trace",
            ),
        )
        return {**state, "researcher_exhausted": True}
    source_hints = state.get("source_hints", [])
    default_route = state.get("model_routes", {}).get("researcher", {})
    researcher_routes = state.get("researcher_routes", []) or [default_route]
    steer_items = runtime.steer_messages.get(state["session_id"], [])
    if steer_items:
        latest = steer_items[-1]
        state = {**state, "focus_query": latest}
        runtime.steer_messages[state["session_id"]] = []
        await _agent_message(state["session_id"], "Human", "Researcher", f"Steer update: {latest}")

    processed = 0
    any_new_query = False
    queries_asked = list(state.get("queries_asked", []))[-200:]
    asked_set = set(str(q) for q in queries_asked if str(q).strip())
    for sub_idx in range(idx, min(idx + batch, len(plan))):
        route = researcher_routes[sub_idx % len(researcher_routes)] if researcher_routes else default_route
        subtask = plan[sub_idx]
        subtask_start = time.time()
        subtask_budget = max(10, int(settings.subtask_time_budget_seconds))
        await runtime.emit(
            state["session_id"],
            _event(
                state["session_id"],
                "Researcher",
                {
                    "subtask_id": subtask.get("id", f"p{sub_idx+1}"),
                    "subtask_index": sub_idx,
                    "research_lane": sub_idx % max(1, len(researcher_routes)),
                    "route_provider": route.get("provider", ""),
                    "route_model": route.get("model", ""),
                },
                "trace",
            ),
        )
        base_query = _build_subtask_query(state["user_query"], subtask)
        await runtime.emit(
            state["session_id"],
            _event(
                state["session_id"],
                "Researcher",
                {
                    "summary": "Selected search lane and retrieval strategy for this subtask.",
                    "base_query": base_query[:220],
                    "lane": sub_idx % max(1, len(researcher_routes)),
                    "route_model": route.get("model", ""),
                },
                "reasoning_summary",
            ),
        )
        query_candidates = [base_query, _clean_query_text(state["user_query"])] + [_clean_query_text(f"{base_query} {h}") for h in source_hints[:2]]
        focus_query = _clean_query_text(state.get("focus_query", "")) if state.get("focus_query") else ""
        if focus_query and focus_query not in (base_query, state["user_query"]) and not _is_meta_query_text(focus_query):
            query_candidates.append(focus_query)
        queries: list[str] = []
        for q in query_candidates:
            cq = _clean_query_text(q)
            if not cq or _is_meta_query_text(cq):
                continue
            if cq in asked_set:
                continue
            queries.append(cq)
            asked_set.add(cq)
            queries_asked.append(cq)
        if not queries:
            await runtime.emit(
                state["session_id"],
                _event(
                    state["session_id"],
                    "Researcher",
                    {
                        "researcher_exhausted": True,
                        "reason": "no_new_queries_after_dedup_or_meta_filter",
                        "base_query": base_query[:200],
                    },
                    "trace",
                ),
            )
            continue
        any_new_query = True

        seen_urls: set[str] = set()
        subtask_docs = 0

        # Budget check before starting parallel search
        if (time.time() - subtask_start) > subtask_budget:
            await runtime.emit(
                state["session_id"],
                _event(
                    state["session_id"],
                    "Researcher",
                    {
                        "subtask_id": subtask.get("id", f"p{sub_idx+1}"),
                        "budget_exhausted": True,
                        "elapsed_sec": round(time.time() - subtask_start, 2),
                        "budget_sec": subtask_budget,
                    },
                    "trace",
                ),
            )
        else:
            # Fan-out: run all search queries concurrently
            _loop = asyncio.get_running_loop()
            search_coros = [
                _loop.run_in_executor(
                    None, web_search, q, state.get("search_results_per_query", 10),
                    settings.searchapi_api_key, settings.tavily_api_key, llm.use_mock,
                )
                for q in queries
            ]
            search_results_raw = await asyncio.gather(*search_coros, return_exceptions=True)

            # Collect unique rows; report failures
            all_rows: list[dict[str, Any]] = []
            for q, res in zip(queries, search_results_raw):
                if isinstance(res, BaseException):
                    fail_row = {"query": q[:220], "provider_order": [], "errors": [str(res)[:220]]}
                elif res.get("status") == "error":
                    fail_row = {
                        "query": q[:220],
                        "provider_order": res.get("payload", {}).get("provider_order", []),
                        "errors": res.get("payload", {}).get("errors", [])[:4],
                    }
                else:
                    fail_row = None
                if fail_row is not None:
                    search_failures.append(fail_row)
                    await runtime.emit(
                        state["session_id"],
                        _event(state["session_id"], "Researcher", {"search_error": fail_row}, "trace"),
                    )
                    continue
                for row in res["payload"]["results"]:
                    if row["url"] not in seen_urls:
                        seen_urls.add(row["url"])
                        all_rows.append(row)

            if all_rows:
                # Fan-out: fetch all unique URLs concurrently (cap at 20)
                url_batch = all_rows[:state.get("url_fetch_limit", 20)]

                async def _fetch_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
                    try:
                        pg = await asyncio.wait_for(
                            _loop.run_in_executor(None, open_url, row["url"], llm.use_mock),
                            timeout=15.0,
                        )
                        if pg.get("status") == "error" or not pg["payload"].get("text"):
                            pg = await asyncio.wait_for(
                                _loop.run_in_executor(None, playwright_fetch, row["url"], 10000),
                                timeout=15.0,
                            )
                    except Exception:
                        pg = {"status": "error", "payload": {"text": ""}}
                    return row, pg

                fetched = await asyncio.gather(*[_fetch_row(r) for r in url_batch], return_exceptions=True)
                tokens = set(t for t in re.sub(r"[^a-z0-9 ]", " ", base_query.lower()).split() if len(t) > 2)

                for fetch_result in fetched:
                    if subtask_docs >= state.get("docs_per_subtask", 8):
                        break
                    if isinstance(fetch_result, Exception):
                        continue
                    row, page = fetch_result
                    if page.get("status") == "error" or not page["payload"].get("text"):
                        snippet = (row.get("snippet") or "").strip()
                        if len(snippet.split()) < 8:
                            continue
                        raw_text = f"{row.get('title', '')}. {snippet}"
                    else:
                        raw_text = page["payload"]["text"]
                    lowered = raw_text.lower()
                    if raw_text.startswith("%PDF") or raw_text.count("\x00") > 0:
                        continue
                    # Only skip if page is dominated by code/CSS (not just mentions it)
                    if lowered.count("var --wp--") > 3 or (lowered.count("function(") > 5 and lowered.count("{") > 20):
                        continue
                    if len(raw_text.split()) < 15:
                        continue

                    summary_txt, claims = await llm.summarize(raw_text, route=route)
                    try:
                        embedding = await embedder.embed(summary_txt)
                    except Exception:
                        embedding = []
                    summary = summarize_text(raw_text) if llm.use_mock else {"trace_id": "llm_summary", "payload": {}}
                    doc = {
                        "session_id": state["session_id"],
                        "source_url": row["url"],
                        "title": row["title"],
                        "timestamp": row["timestamp"],
                        "summary": summary_txt,
                        "claims": claims,
                        "embedding": embedding,
                        "verification_score": 0.0,
                        "contradiction": False,
                        "trace_id": summary.get("trace_id", "llm_summary"),
                    }
                    qdrant_upsert(doc)
                    snapshot_url(row["url"], state["session_id"])
                    evidence.append(doc)
                    subtask_docs += 1
                    await runtime.emit(
                        state["session_id"],
                        _event(state["session_id"], "Researcher", doc, "evidence_item"),
                    )

        if subtask_docs == 0:
            # Generic fallback: direct Wikipedia summary lookup for the subquestion.
            wiki = wikipedia_lookup(base_query)
            if wiki.get("status") == "ok" and wiki["payload"].get("text"):
                raw_text = wiki["payload"]["text"]
                summary_txt, claims = await llm.summarize(raw_text, route=route)
                try:
                    embedding = await embedder.embed(summary_txt)
                except Exception:
                    embedding = []
                doc = {
                    "session_id": state["session_id"],
                    "source_url": wiki["payload"]["url"],
                    "title": wiki["payload"]["title"],
                    "timestamp": str(int(time.time())),
                    "summary": summary_txt,
                    "claims": claims,
                    "embedding": embedding,
                    "verification_score": 0.0,
                    "contradiction": False,
                    "trace_id": "wiki_fallback",
                }
                qdrant_upsert(doc)
                snapshot_url(wiki["payload"]["url"], state["session_id"])
                evidence.append(doc)
                subtask_docs += 1
                await runtime.emit(
                    state["session_id"],
                    _event(state["session_id"], "Researcher", doc, "evidence_item"),
                )

        if subtask_docs > 0:
            subtask["status"] = "done"
            processed += 1

    await runtime.emit(
        state["session_id"],
        _event(
            state["session_id"],
            "Researcher",
            {"processed_subtasks": processed, "batch": batch, "evidence_items_total": len(evidence)},
            "trace",
        ),
    )
    exhausted = (not any_new_query) or (processed == 0 and idx >= len(plan) - 1 and len(evidence) == start_evidence_count)
    if exhausted:
        await runtime.emit(
            state["session_id"],
            _event(
                state["session_id"],
                "Researcher",
                {"researcher_exhausted": True, "reason": "no_progress_in_cycle", "processed_subtasks": processed},
                "trace",
            ),
        )
    return {
        **state,
        "evidence_items": evidence,
        "current_plan_index": idx + processed,
        "search_failures": search_failures,
        "queries_asked": queries_asked,
        "researcher_exhausted": exhausted,
    }


async def _verify_one_item(
    item: dict[str, Any],
    query: str,
    passes: int,
    route: dict[str, Any],
) -> dict[str, Any]:
    """Verify a single evidence item — single pass, no re-fetch.

    The researcher already fetched and summarized the content. Re-fetching
    the same URL wastes time and network. We verify against the summary
    and claims the researcher already extracted.
    """
    # Use the summary as source context — no re-fetch needed
    source_excerpt = item.get("summary", "")

    try:
        s, c, r = await llm.verify(
            query=query,
            title=item.get("title", ""),
            summary=item.get("summary", ""),
            claims=item.get("claims", []),
            source_excerpt=source_excerpt,
            route=route,
        )
    except Exception as exc:
        s, c, r = 0.45, False, f"verifier_error_fallback: {str(exc)[:220]}"

    return {**item, "verification_score": max(0.0, min(1.0, s)), "contradiction": c, "verification_note": r}


async def verifier_node(state: SwarmState) -> SwarmState:
    _routes = state.get("model_routes", {})
    route = _routes.get("verifier") or _routes.get("analyst") or {}
    passes = max(1, int(state.get("verifier_passes", 1)))
    items = state.get("evidence_items", [])
    query = state.get("user_query", "")
    focus_raw = state.get("verifier_focus_indices", [])
    focus: list[int] = []
    if isinstance(focus_raw, list):
        for idx in focus_raw:
            if isinstance(idx, (int, float)):
                i = int(idx)
                if 0 <= i < len(items):
                    focus.append(i)
    target_indices = sorted(set(focus)) if focus else list(range(len(items)))
    target_items = [items[i] for i in target_indices]

    scored_raw = await asyncio.gather(
        *[asyncio.wait_for(_verify_one_item(item, query, passes, route), timeout=60.0) for item in target_items],
        return_exceptions=True,
    )
    scored_targets: list[dict[str, Any]] = []
    for item, result in zip(target_items, scored_raw):
        if isinstance(result, Exception):
            scored_targets.append({**item, "verification_score": 0.25, "contradiction": False, "verification_note": f"timeout/error: {str(result)[:100]}"})
        else:
            scored_targets.append(result)
    scored = list(items)
    for i, upd in zip(target_indices, scored_targets):
        scored[i] = upd

    contradictions = sum(1 for i in scored if i.get("contradiction"))
    avg_score = (
        sum(float(i.get("verification_score", 0.0)) for i in scored) / len(scored)
        if scored
        else 0.0
    )
    verification = {
        "items": scored,
        "avg_verification_score": avg_score,
        "contradictions": contradictions,
        "route_model": route.get("model", ""),
        "scope": target_indices,
    }
    event = _event(state["session_id"], "Verifier", verification, "trace")
    await runtime.emit(state["session_id"], event)
    weak = [i for i in scored if float(i.get("verification_score", 0.0)) < 0.6]
    await runtime.emit(
        state["session_id"],
        _event(
            state["session_id"],
            "Verifier",
            {
                "summary": "Scored evidence for support/contradiction and selected confidence bounds.",
                "avg_score": verification.get("avg_verification_score", 0.0),
                "weak_items": len(weak),
                "contradictions": contradictions,
                "route_model": route.get("model", ""),
            },
            "reasoning_summary",
        ),
    )
    await _agent_message(state["session_id"], "Verifier", "Analyst", "Verification pass complete. Use supported claims and flag weak evidence.")
    return {**state, "verification": verification, "evidence_items": verification["items"], "verifier_focus_indices": []}


async def analyst_node(state: SwarmState) -> SwarmState:
    items = state.get("evidence_items", [])
    default_score = settings.analyst_good_score_factoid if state.get("task_type") == "factoid" else settings.analyst_good_score
    
    # Dynamic Threshold Decay: Lower standards as cycles increase to force convergence
    cycle_count = state.get("cycle_count", 0) + 1
    decay_factor = max(0.0, (cycle_count - 1) * 0.05)
    min_score = max(0.35, float(state.get("analyst_min_score") or default_score) - decay_factor)
    
    good = [i for i in items if i.get("verification_score", 0) >= min_score and not i.get("contradiction")]
    provisional = sorted(items, key=lambda x: x.get("verification_score", 0), reverse=True)[:3]
    used = max(1, len(good))
    context_precision = used / max(1, len(items))
    cycle_count = state.get("cycle_count", 0) + 1
    min_needed = max(1, int(state.get("min_verified_evidence", 2)))
    pending_plan = state.get("current_plan_index", 0) < len(state.get("plan", []))
    # Only force more research if we have zero usable evidence AND pending subtasks
    heuristic_needs_more = len(good) == 0 and pending_plan and cycle_count < settings.max_cycles
    _routes = state.get("model_routes", {})
    route = _routes.get("analyst") or _routes.get("verifier") or {}
    decision = await llm.analyst_decide(
        query=state.get("user_query", ""),
        evidence_items=items,
        cycle_count=cycle_count,
        max_cycles=settings.max_cycles,
        min_verified_evidence=min_needed,
        route=route,
    )
    # Trust the LLM decision primarily; heuristic only kicks in when evidence is empty
    needs_more = bool(decision.get("needs_more_research", False)) or heuristic_needs_more
    if cycle_count >= settings.max_cycles:
        needs_more = False
    focus_query = str(decision.get("focus_query", "")).strip()
    ask_user = bool(decision.get("ask_user", False))
    question = str(decision.get("question", "")).strip()
    rationale = str(decision.get("rationale", "")).strip()
    missing_requirements = missing_requirement_facets(state.get("user_query", ""), items)
    if ask_user and question and _should_ask_hitl(state, "analysis", question):
        runtime.hitl_questions[state["session_id"]] = question
        ask = ask_user_question(question)
        await runtime.emit(state["session_id"], _event(state["session_id"], "HITL_Analyst", ask["payload"], "hitl_request"))
    else:
        ask_user = False
        question = ""

    payload = {
        "context_precision": round(context_precision, 3),
        "cycle_count": cycle_count,
        "needs_more_research": needs_more,
    }
    await runtime.emit(state["session_id"], _event(state["session_id"], "Analyst", payload, "trace"))
    await runtime.emit(
        state["session_id"],
        _event(
            state["session_id"],
            "Analyst",
            {
                "summary": "Decided whether to continue research or synthesize final output.",
                "needs_more_research": needs_more,
                "focus_query": focus_query,
                "rationale": rationale[:320],
                "route_model": route.get("model", ""),
            },
            "reasoning_summary",
        ),
    )
    await _agent_message(state["session_id"], "Analyst", "Writer", "Synthesis directive ready. Separate facts/inference/hypothesis in final output.")
    return {
        **state,
        "final_evidence_set": good,
        "provisional_evidence_set": provisional,
        "missing_requirements": missing_requirements,
        "needs_more_research": needs_more,
        "cycle_count": cycle_count,
        "focus_query": focus_query,
        "pending_hitl": ask_user and bool(question),
        "interrupt_stage": "analysis" if ask_user and bool(question) else "",
        "hitl_question": question if ask_user and bool(question) else state.get("hitl_question", ""),
        "hitl_last_stage": "analysis" if ask_user and bool(question) else state.get("hitl_last_stage", ""),
        "hitl_last_question": question if ask_user and bool(question) else state.get("hitl_last_question", ""),
    }


async def writer_node(state: SwarmState) -> SwarmState:
    evidence = state.get("final_evidence_set", [])
    search_failures = state.get("search_failures", []) or []
    if not evidence:
        evidence = state.get("provisional_evidence_set", [])
        if not evidence:
            # No evidence found: surface concrete retrieval failure, do not hallucinate citations.
            failure_note = "External retrieval failed; no web evidence could be collected."
            if search_failures:
                first = search_failures[0]
                errs = first.get("errors", [])
                if errs:
                    failure_note += " First errors: " + "; ".join(str(e) for e in errs[:2])
            evidence = [{"summary": failure_note, "source_url": "local://search-failure"}]
        await runtime.emit(
            state["session_id"],
            _event(
                state["session_id"],
                "Writer",
                {"warning": "Using provisional evidence due to low verification coverage."},
                "trace",
            ),
        )
    route = state.get("model_routes", {}).get("writer", {})
    draft, final = await llm.write(state["user_query"], evidence, route=route, sections=state.get("writer_sections") or [])
    missing_requirements = state.get("missing_requirements", []) or []
    diagnostics: dict[str, Any] = {}
    if search_failures:
        diagnostics["search_failures"] = [
            {
                "query": str(row.get("query", ""))[:160],
                "errors": [str(e)[:180] for e in row.get("errors", [])[:3]],
            }
            for row in search_failures[:5]
        ]
    if missing_requirements:
        diagnostics["missing_requirements"] = [str(x)[:120] for x in missing_requirements[:20]]
    if diagnostics:
        await runtime.emit(
            state["session_id"],
            _event(state["session_id"], "Writer", diagnostics, "trace"),
        )
    plan = state.get("plan", [])
    for item in plan:
        item["status"] = "done"
    await runtime.emit(state["session_id"], _event(state["session_id"], "Writer", {"draft": draft}, "draft_markdown"))
    await runtime.emit(state["session_id"], _event(state["session_id"], "Writer", {"final": final}, "final_markdown"))
    await runtime.emit(
        state["session_id"],
        _event(
            state["session_id"],
            "Writer",
            {
                "summary": "Synthesized report from verified/provisional evidence with citations." if evidence else "Generating search failure report (no evidence found).",
                "evidence_used": len(evidence),
                "output_chars": len(final),
                "route_model": route.get("model", ""),
            },
            "reasoning_summary",
        ),
    )
    return {**state, "draft_markdown": draft, "final_markdown": final, "plan": plan, "writer_diagnostics": diagnostics}


async def hitl_review_node(state: SwarmState) -> SwarmState:
    session_id = state["session_id"]
    answer = runtime.hitl_answers.pop(session_id, None)
    if answer:
        cleaned = answer.strip()
        if cleaned.lower() in _HITL_APPROVAL_KEYWORDS or _is_defer_answer(cleaned):
            await runtime.emit(session_id, _event(session_id, "HITL_Review", {"answer": cleaned}, "trace"))
            skipped = set(state.get("hitl_opt_out_stages", []))
            if _is_defer_answer(cleaned):
                skipped.add("review")
            return {
                **state,
                "pending_hitl": False,
                "needs_revision": False,
                "review_feedback": cleaned,
                "hitl_opt_out_stages": sorted(skipped),
                "autonomous_mode": True if _is_defer_answer(cleaned) else state.get("autonomous_mode", False),
            }
        await runtime.emit(session_id, _event(session_id, "HITL_Review", {"answer": cleaned}, "trace"))
        return {
            **state,
            "pending_hitl": False,
            "needs_revision": True,
            "review_feedback": cleaned,
            "focus_query": cleaned,
        }

    route = state.get("model_routes", {}).get("writer", {})
    decision = await llm.decide_hitl(
        stage="review",
        query=state.get("user_query", ""),
        context=state.get("draft_markdown", "")[:2500],
        route=route,
    )
    question = str(decision.get("question", "")).strip()
    if decision.get("ask_user") and question and _should_ask_hitl(state, "review", question):
        ask = ask_user_question(question)
        await runtime.emit(session_id, _event(session_id, "HITL_Review", ask["payload"], "hitl_request"))
        return {
            **state,
            "pending_hitl": True,
            "hitl_question": question,
            "interrupt_stage": "review",
            "hitl_last_stage": "review",
            "hitl_last_question": question,
        }
    return {**state, "pending_hitl": False, "interrupt_stage": "", "review_feedback": "auto-continue"}


async def postprocess_node(state: SwarmState) -> SwarmState:
    retrieval = qdrant_search(state["session_id"], state["user_query"], limit=5)
    payload = {"citation_count": len(retrieval["payload"]["results"])}
    await runtime.emit(state["session_id"], _event(state["session_id"], "PostProcess", payload, "trace"))
    return {**state, "postprocessed": True}


async def save_node(state: SwarmState) -> SwarmState:
    report_dir = Path(settings.report_dir) / state["session_id"]
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"
    report_path.write_text(state.get("final_markdown", ""), encoding="utf-8")
    exports = maybe_export_artifacts(state["session_id"], state.get("user_query", ""), state.get("final_markdown", ""))
    
    # Neural Cartographer: Generate Semantic Knowledge Map
    if state.get("evidence_items"):
        canvas_path = await generate_knowledge_map(
            state["session_id"],
            state["user_query"],
            state.get("evidence_items", []),
            str(report_dir),
            llm=llm,
        )
        if canvas_path:
            exports.setdefault("generated", {})["knowledge_map"] = canvas_path

    await runtime.emit(
        state["session_id"],
        _event(
            state["session_id"],
            "SaveArtifact",
            {"path": str(report_path), "artifacts": exports.get("generated", {}), "artifact_warnings": exports.get("warnings", [])},
            "trace",
        ),
    )
    return {**state, "artifacts": exports.get("generated", {}), "save_done": True}


async def metrics_node(state: SwarmState) -> SwarmState:
    plan = state.get("plan", [])
    done = [p for p in plan if p.get("status") == "done"]
    executed_ratio = len(done) / max(1, len(plan))
    context_precision = len(state.get("final_evidence_set", [])) / max(1, len(state.get("evidence_items", [])))
    faithfulness = state.get("verification", {}).get("avg_verification_score", 0.0)
    metrics = {
        "task_completion": 1.0 if state.get("final_markdown") else 0.0,
        "plan_quality": 0.8,
        "plan_adherence": round(executed_ratio, 3),
        "tool_correctness": 1.0,
        "argument_correctness": 1.0,
        "rag_faithfulness": round(float(faithfulness), 3),
        "context_precision": round(context_precision, 3),
        "step_efficiency": round(len(plan) / max(1, state.get("cycle_count", 1)), 3),
    }
    await runtime.emit(state["session_id"], _event(state["session_id"], "MetricsEmitter", metrics, "trace"))
    return {**state, "metrics": metrics}


def _orchestrator_router(state: SwarmState) -> str:
    nxt = str(state.get("orchestrator_next", "researcher"))
    valid = {
        "planner",
        "researcher",
        "verifier",
        "analyst",
        "writer",
        "hitl",
        "hitl_review",
        "postprocess",
        "save",
        "metrics",
        "end_waiting",
        "end",
    }
    return nxt if nxt in valid else "researcher"


def _with_last_worker(name: str, fn):
    async def _wrapped(state: SwarmState) -> SwarmState:
        out = await fn(state)
        return {**out, "last_worker": name}

    return _wrapped


def _build_graph():
    graph = StateGraph(SwarmState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("planner", _with_last_worker("planner", planner_node))
    graph.add_node("hitl", _with_last_worker("hitl", hitl_node))
    graph.add_node("researcher", _with_last_worker("researcher", researcher_node))
    graph.add_node("verifier", _with_last_worker("verifier", verifier_node))
    graph.add_node("analyst", _with_last_worker("analyst", analyst_node))
    graph.add_node("writer", _with_last_worker("writer", writer_node))
    graph.add_node("hitl_review", _with_last_worker("hitl_review", hitl_review_node))
    graph.add_node("postprocess", _with_last_worker("postprocess", postprocess_node))
    graph.add_node("save", _with_last_worker("save", save_node))
    graph.add_node("metrics", _with_last_worker("metrics", metrics_node))

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        _orchestrator_router,
        {
            "planner": "planner",
            "researcher": "researcher",
            "verifier": "verifier",
            "analyst": "analyst",
            "writer": "writer",
            "hitl": "hitl",
            "hitl_review": "hitl_review",
            "postprocess": "postprocess",
            "save": "save",
            "metrics": "metrics",
            "end_waiting": END,
            "end": END,
        },
    )

    graph.add_edge("planner", "orchestrator")
    graph.add_edge("hitl", "orchestrator")
    graph.add_edge("researcher", "orchestrator")
    graph.add_edge("verifier", "orchestrator")
    graph.add_edge("analyst", "orchestrator")
    graph.add_edge("writer", "orchestrator")
    graph.add_edge("hitl_review", "orchestrator")
    graph.add_edge("postprocess", "orchestrator")
    graph.add_edge("save", "orchestrator")
    graph.add_edge("metrics", "orchestrator")

    checkpointer = None
    enable_sqlite = os.getenv("ENABLE_SQLITE_CHECKPOINT", "").strip().lower() in {"1", "true", "yes", "on"}
    if enable_sqlite and SqliteSaver:
        try:
            Path(settings.checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)
            checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{settings.checkpoint_db_path}")
        except Exception:
            checkpointer = None
    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    try:
        return graph.compile(checkpointer=InMemorySaver())
    except Exception:
        return graph.compile()


compiled_graph = _build_graph()


async def run_session(
    session_id: str,
    user_query: str,
    swarm_cfg_override: SwarmConfig | None = None,
    isolated_graph: bool = False,
    cross_session_memory: bool = False,
    autonomous_mode: bool = False,
    budget_mode: str = "balanced",
    depth: str = "standard",
    provider_pref: str = "auto",
    preference_text: str = "",
) -> None:
    langsmith_env(settings.langsmith_api_key)
    logger.info("run_session_start", extra={"event": "run_session_start", "session_id": session_id})
    runtime.status[session_id] = "running"
    supabase_store.upsert_session(session_id, "running")
    swarm_cfg = swarm_cfg_override or build_swarm_config(user_query, budget_mode=budget_mode, depth=depth, provider_pref=provider_pref)
    if not cross_session_memory:
        mem_boot = {"status": "skipped", "payload": {"reason": "cross-session memory disabled"}}
        mem_focus = {"status": "skipped", "payload": {"results": []}}
    elif llm.use_mock:
        mem_boot = {"status": "skipped", "payload": {"reason": "mock llm"}}
        mem_focus = {"status": "skipped", "payload": {"results": []}}
    else:
        mem_boot = mem_bootstrap_pack(user_query)
        mem_focus = mem_search_pack(user_query, limit=6)
    mem_lines: list[str] = []
    for row in mem_focus.get("payload", {}).get("results", [])[:4]:
        snippet = str(row.get("snippet", "")).strip()
        if snippet:
            mem_lines.append(f"- {snippet}")
    memory_context = "\n".join(mem_lines)
    await runtime.emit(
        session_id,
        _event(
            session_id,
            "Memory",
            {
                "bootstrap_status": mem_boot.get("status"),
                "search_status": mem_focus.get("status"),
                "cross_session_memory": cross_session_memory,
                "recalled_items": len(mem_focus.get("payload", {}).get("results", []) if isinstance(mem_focus.get("payload", {}), dict) else []),
            },
            "trace",
        ),
    )
    state: SwarmState = {
        "session_id": session_id,
        "user_query": user_query,
        "evidence_items": [],
        "trace": [],
        "tool_logs": [],
        "budget_mode": budget_mode,
        "depth": depth,
        "provider_pref": provider_pref,
        "task_type": swarm_cfg.task_type,
        "complexity": swarm_cfg.complexity,
        "research_batch_size": 1,
        "verifier_passes": 1,
        "min_verified_evidence": 1,
        "source_hints": swarm_cfg.source_hints,
        "model_routes": swarm_cfg.model_routes,
        "researcher_routes": swarm_cfg.researcher_routes,
        "memory_context": memory_context,
        "cross_session_memory": cross_session_memory,
        "autonomous_mode": autonomous_mode,
        "hitl_opt_out_stages": ["plan", "analysis", "review"] if autonomous_mode else [],
        "plan_approved": False,
        "interrupt_stage": "",
        "tool_registry": tool_registry(),
        "orchestration_history": [],
        "orchestrator_next": "planner",
        "last_worker": "",
        "budget_steps_remaining": _step_budget_for_mode(budget_mode),
        "budget_steps_used": 0,
        "budget_exhausted": False,
        "postprocessed": False,
        "save_done": False,
        "force_replan": False,
        "verifier_focus_indices": [],
        "post_final_iterations": 0,
        "queries_asked": [],
        "researcher_exhausted": False,
        "replan_attempts": 0,
    }
    cfg = {"configurable": {"thread_id": session_id}}
    graph_runner = _build_graph() if isolated_graph else compiled_graph
    try:
        result = await graph_runner.ainvoke(state, config=cfg)
        runtime.final_state[session_id] = result
        runtime.status[session_id] = "waiting_hitl" if result.get("pending_hitl") else "completed"
        logger.info("run_session_done", extra={"event": "run_session_done", "session_id": session_id})
        if not llm.use_mock:
            mem_save_note(
                content=f"Session {session_id} completed. Query: {user_query}. Selected summary: {result.get('final_markdown','')[:1200]}",
                tags=["session", "result", "research-swarm"],
                importance=3,
            )
        supabase_store.upsert_session(session_id, runtime.status[session_id])
        await runtime.emit(session_id, {"type": "done", "payload": {"status": runtime.status[session_id]}})
    except Exception as exc:
        runtime.status[session_id] = "failed"
        runtime.final_state[session_id] = {**state, "error": str(exc)}
        logger.exception("run_session_failed", extra={"event": "run_session_failed", "session_id": session_id})
        if not llm.use_mock:
            mem_save_note(
                content=f"Session {session_id} failed. Query: {user_query}. Error: {str(exc)[:800]}",
                tags=["session", "failure", "research-swarm"],
                importance=4,
            )
        supabase_store.upsert_session(session_id, "failed")
        await runtime.emit(session_id, {"type": "done", "payload": {"status": "failed", "error": str(exc)}})


async def _forward_lane_events(parent_session_id: str, child_session_id: str, lane_id: str) -> None:
    while True:
        event = await runtime.queues[child_session_id].get()
        # Inject lane_id into payload for visibility
        if "payload" in event and isinstance(event["payload"], dict):
            event["payload"]["lane_id"] = lane_id
            
        wrapped = {
            "type": "portfolio_lane_event",
            "lane_id": lane_id,
            "child_session_id": child_session_id,
            "event": event,
            "node_name": event.get("node_name", "lane"),
            "payload": event.get("payload", {}),
        }
        await runtime.emit(parent_session_id, wrapped)
        if event.get("type") == "done":
            break


async def _broadcast_steer(parent_session_id: str, active_children: set[str], stop_evt: asyncio.Event) -> None:
    while not stop_evt.is_set():
        await asyncio.sleep(0.2)
        messages = runtime.steer_messages.get(parent_session_id, [])
        if not messages:
            continue
        runtime.steer_messages[parent_session_id] = []
        for child in list(active_children):
            runtime.steer_messages[child].extend(messages)
        await runtime.emit(
            parent_session_id,
            _event(
                parent_session_id,
                "HITL_Steer",
                {"broadcast_to": len(active_children), "messages": messages},
                "trace",
            ),
        )


async def run_portfolio_session(
    session_id: str,
    user_query: str,
    prefs: PortfolioPreferences,
    cross_session_memory: bool = False,
) -> None:
    langsmith_env(settings.langsmith_api_key)
    logger.info("run_portfolio_start", extra={"event": "run_portfolio_start", "session_id": session_id})
    runtime.status[session_id] = "running"
    supabase_store.upsert_session(session_id, "running")
    lanes = build_portfolio_lanes(user_query, prefs)
    runtime.portfolio_children[session_id] = []

    await runtime.emit(
        session_id,
        _event(
            session_id,
            "PortfolioPlanner",
            {
                "lane_count": len(lanes),
                "lanes": [
                    {
                        "lane_id": l.lane_id,
                        "label": l.label,
                        "provider": l.provider,
                        "strategy": l.strategy,
                        "models": {
                            "planner": l.config.model_routes.get("planner", {}).get("model", ""),
                            "researcher": l.config.model_routes.get("researcher", {}).get("model", ""),
                            "analyst": l.config.model_routes.get("analyst", {}).get("model", ""),
                            "verifier": l.config.model_routes.get("verifier", {}).get("model", ""),
                            "writer": l.config.model_routes.get("writer", {}).get("model", ""),
                            "researcher_pool": [r.get("model", "") for r in l.config.researcher_routes if r.get("model")],
                        },
                    }
                    for l in lanes
                ],
                "auto_orchestration": {
                    "provider_order": provider_order(prefs),
                    "budget_mode": prefs.budget_mode,
                    "depth": prefs.depth,
                    "premium_intent": prefs.budget_mode == "high" or "no problem with money" in prefs.preference_text.lower(),
                },
                "preferences": {
                    "budget_mode": prefs.budget_mode,
                    "depth": prefs.depth,
                    "provider_pref": prefs.provider_pref,
                    "detail_level": prefs.detail_level,
                    "preference_text": prefs.preference_text,
                    "cross_session_memory": cross_session_memory,
                },
            },
            "portfolio_plan",
        ),
    )

    active_children: set[str] = set()
    stop_evt = asyncio.Event()
    steer_task = asyncio.create_task(_broadcast_steer(session_id, active_children, stop_evt))

    async def run_lane(lane: Any) -> dict[str, Any]:
        child_session_id = f"{session_id}_{lane.lane_id}"
        runtime.portfolio_children[session_id].append(child_session_id)
        active_children.add(child_session_id)
        start = time.time()
        await runtime.emit(
            session_id,
            _event(
                session_id,
                "PortfolioLane",
                {
                    "lane_id": lane.lane_id,
                    "label": lane.label,
                    "provider": lane.provider,
                    "strategy": lane.strategy,
                    "child_session_id": child_session_id,
                },
                "portfolio_lane_start",
            ),
        )
        forward_task = asyncio.create_task(_forward_lane_events(session_id, child_session_id, lane.lane_id))
        try:
            await run_session(
                child_session_id,
                user_query,
                swarm_cfg_override=lane.config,
                isolated_graph=True,
                cross_session_memory=cross_session_memory,
                autonomous_mode=True,
            )
            loops = 0
            start_wait = time.time()
            while runtime.status.get(child_session_id) == "waiting_hitl" and loops < 2 and (time.time() - start_wait) < 30:
                await resume_session(child_session_id, "approve")
                loops += 1
            if runtime.status.get(child_session_id) == "waiting_hitl":
                runtime.status[child_session_id] = "failed"
                runtime.final_state[child_session_id] = {
                    **runtime.final_state.get(child_session_id, {}),
                    "error": "portfolio lane paused for HITL (autonomous mode should prevent this; check route prompts)",
                }
                await runtime.emit(child_session_id, {"type": "done", "payload": {"status": "failed", "error": "hitl_limit"}})
        except Exception as exc:
            runtime.status[child_session_id] = "failed"
            runtime.final_state[child_session_id] = {
                **runtime.final_state.get(child_session_id, {}),
                "session_id": child_session_id,
                "error": f"lane_runtime_error: {str(exc)[:220]}",
            }
            await runtime.emit(child_session_id, {"type": "done", "payload": {"status": "failed", "error": str(exc)}})
        finally:
            try:
                await forward_task
            except Exception:
                pass
        latency = max(0.001, time.time() - start)
        child_state = runtime.final_state.get(child_session_id, {})
        try:
            score_vec = compute_candidate_metrics(user_query, lane, child_state, latency_s=latency)
        except Exception as exc:
            score_vec = {
                "faithfulness": 0.0,
                "coverage": 0.0,
                "recency": 0.0,
                "novelty": 0.0,
                "cost": 0.0,
                "latency": round(latency, 2),
            }
            child_state = {**child_state, "error": f"candidate_metric_error: {str(exc)[:180]}"}
        status = runtime.status.get(child_session_id, "failed")
        # If a lane produced a final report but later failed on teardown/post-step, keep it usable.
        usable = bool(child_state.get("final_markdown")) and status != "waiting_hitl"
        candidate = {
            "lane_id": lane.lane_id,
            "label": lane.label,
            "provider": lane.provider,
            "strategy": lane.strategy,
            "session_id": child_session_id,
            "status": status,
            "usable": usable,
            "score_vector": score_vec,
            "model_routes": child_state.get("model_routes", lane.config.model_routes),
            "final_markdown": child_state.get("final_markdown", ""),
            "metrics": child_state.get("metrics", {}),
            "error": str(child_state.get("error", "") or ""),
            "latency_seconds": latency,
            "quality": {
                "faithfulness": qualitative_band(score_vec["faithfulness"]),
                "coverage": qualitative_band(score_vec["coverage"]),
                "recency": qualitative_band(score_vec["recency"]),
                "novelty": qualitative_band(score_vec["novelty"]),
            },
        }
        await runtime.emit(
            session_id,
            _event(
                session_id,
                "PortfolioLane",
                {
                    "lane_id": lane.lane_id,
                    "status": candidate["status"],
                    "usable": usable,
                    "score_vector": score_vec,
                    "quality": candidate["quality"],
                    "latency_seconds": round(latency, 2),
                    "child_session_id": child_session_id,
                    "error": candidate["error"],
                },
                "portfolio_lane_done",
            ),
        )
        active_children.discard(child_session_id)
        return candidate

    try:
        candidates = await asyncio.gather(*(run_lane(l) for l in lanes))
        usable = [c for c in candidates if c.get("usable", False)]
        try:
            frontier = pareto_frontier(usable)
        except Exception:
            frontier = []
        try:
            chosen = choose_candidate(frontier, prefs) or (frontier[0] if frontier else None)
        except Exception:
            chosen = frontier[0] if frontier else None
        chosen_markdown = chosen.get("final_markdown", "") if chosen else ""
        exports = maybe_export_artifacts(session_id, user_query, chosen_markdown)
        portfolio_state = {
            "session_id": session_id,
            "mode": "portfolio",
            "user_query": user_query,
            "portfolio_candidates": candidates,
            "pareto_frontier_lane_ids": [c["lane_id"] for c in frontier],
            "selected_lane_id": chosen.get("lane_id") if chosen else "",
            "selected_session_id": chosen.get("session_id") if chosen else "",
            "selected_score_vector": chosen.get("score_vector", {}) if chosen else {},
            "selected_model_routes": chosen.get("model_routes", {}) if chosen else {},
            "final_markdown": chosen_markdown,
            "artifacts": exports.get("generated", {}),
            "portfolio_preferences": {
                "budget_mode": prefs.budget_mode,
                "depth": prefs.depth,
                "provider_pref": prefs.provider_pref,
                "detail_level": prefs.detail_level,
                "preference_text": prefs.preference_text,
                "cross_session_memory": cross_session_memory,
            },
            "metrics": chosen.get("metrics", {}) if chosen else {},
            "error": "" if chosen else "No candidate completed.",
        }
        runtime.final_state[session_id] = portfolio_state
        runtime.status[session_id] = "completed" if chosen else "failed"
        logger.info("run_portfolio_done", extra={"event": "run_portfolio_done", "session_id": session_id})
        await runtime.emit(
            session_id,
            _event(
                session_id,
                "PortfolioSummary",
                {
                    "candidates": len(candidates),
                    "usable_candidates": len(usable),
                    "frontier": [c["lane_id"] for c in frontier],
                    "selected": portfolio_state.get("selected_lane_id", ""),
                    "artifacts": exports.get("generated", {}),
                },
                "portfolio_summary",
            ),
        )
        supabase_store.upsert_session(session_id, runtime.status[session_id])
        await runtime.emit(session_id, {"type": "done", "payload": {"status": runtime.status[session_id]}})
    except Exception as exc:
        runtime.status[session_id] = "failed"
        runtime.final_state[session_id] = {"session_id": session_id, "mode": "portfolio", "error": str(exc), "user_query": user_query}
        logger.exception("run_portfolio_failed", extra={"event": "run_portfolio_failed", "session_id": session_id})
        supabase_store.upsert_session(session_id, "failed")
        await runtime.emit(session_id, {"type": "done", "payload": {"status": "failed", "error": str(exc)}})
    finally:
        stop_evt.set()
        try:
            await steer_task
        except Exception:
            pass


async def resume_session(session_id: str, answer: str) -> None:
    runtime.hitl_answers[session_id] = answer
    runtime.status[session_id] = "running"
    supabase_store.upsert_session(session_id, "running")
    cfg = {"configurable": {"thread_id": session_id}}
    prior = runtime.final_state.get(session_id, {"session_id": session_id})
    resumed = {**prior, "pending_hitl": False}
    try:
        result = await compiled_graph.ainvoke(resumed, config=cfg)
        runtime.final_state[session_id] = result
        runtime.status[session_id] = "waiting_hitl" if result.get("pending_hitl") else "completed"
        supabase_store.upsert_session(session_id, runtime.status[session_id])
        await runtime.emit(session_id, {"type": "done", "payload": {"status": runtime.status[session_id]}})
    except Exception as exc:
        runtime.status[session_id] = "failed"
        runtime.final_state[session_id] = {**resumed, "error": str(exc)}
        supabase_store.upsert_session(session_id, "failed")
        await runtime.emit(session_id, {"type": "done", "payload": {"status": "failed", "error": str(exc)}})


def new_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:12]}"
