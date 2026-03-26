from __future__ import annotations

import atexit
import itertools
import json
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field
from typing import Any

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

_ICONS = {
    "Planner":       "◆",
    "Orchestrator":  "⊛",
    "Researcher":    "◈",
    "Verifier":      "◎",
    "Analyst":       "◉",
    "Writer":        "✦",
    "AgentHub":      "⟡",
    "HITL_Approval": "?",
    "HITL_Analyst":  "?",
    "HITL_Review":   "?",
    "SaveArtifact":  "↓",
    "MetricsEmitter": "~",
    "PostProcess":   "~",
}

_AGENT_COLOR = {
    "Planner":       CYAN,
    "Orchestrator":  CYAN,
    "Researcher":    GREEN,
    "Verifier":      YELLOW,
    "Analyst":       BLUE,
    "Writer":        MAGENTA,
}

_BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@dataclass
class UIState:
    pending_hitl: str = ""
    session_state: str = "running"
    evidence_count: int = 0
    active_agents: set[str] = field(default_factory=set)


_STATE = UIState()
_SESSION_START = time.time()

# Spinner state
_SPINNER_STOP = threading.Event()
_SPINNER_THREAD: threading.Thread | None = None
_SPINNER_TEXT = ["thinking"]
_SPINNER_LOCK = threading.Lock()


def get_ui() -> UIState:
    return _STATE


def wrap(text: str, width: int = 100) -> str:
    return "\n".join(textwrap.wrap(text, width=width)) if text else ""


def quality_word(q: str) -> str:
    if q == "high":
        return "strong"
    if q == "medium":
        return "solid"
    if q == "low":
        return "weak"
    return "n/a"


def _clear_line() -> None:
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def _spin_loop() -> None:
    for ch in itertools.cycle(_BRAILLE):
        if _SPINNER_STOP.is_set():
            break
        with _SPINNER_LOCK:
            txt = _SPINNER_TEXT[0]
        sys.stdout.write(f"\r  {DIM}{ch} {txt}{RESET}")
        sys.stdout.flush()
        time.sleep(0.08)
    _clear_line()


def start_spinner(text: str = "thinking") -> None:
    global _SPINNER_THREAD
    stop_spinner()
    with _SPINNER_LOCK:
        _SPINNER_TEXT[0] = text
    _SPINNER_STOP.clear()
    _SPINNER_THREAD = threading.Thread(target=_spin_loop, daemon=True)
    _SPINNER_THREAD.start()


def stop_spinner() -> None:
    global _SPINNER_THREAD
    _SPINNER_STOP.set()
    if _SPINNER_THREAD and _SPINNER_THREAD.is_alive():
        _SPINNER_THREAD.join(timeout=0.3)
    _SPINNER_THREAD = None
    _SPINNER_STOP.clear()
    _clear_line()


def shutdown_tui() -> None:
    stop_spinner()


def _emit(line: str, color: str = "") -> None:
    stop_spinner()
    if color:
        print(f"{color}{line}{RESET}")
    else:
        print(line)
    sys.stdout.flush()


def render_cyberpunk_banner() -> None:
    lines = [
        "  _   _                      ",
        " | \\ | | ___ _   _ _ __ ___ ",
        " |  \\| |/ _ \\ | | | '__/ _ \\",
        " | |\\  |  __/ |_| | | | (_) |",
        " |_| \\_|\\___|\\__,_|_|  \\___/ ",
        "  Research Swarm",
    ]
    for i, row in enumerate(lines):
        c = MAGENTA if i < 5 else CYAN
        print(f"{BOLD}{c}{row}{RESET}")
    print()


def _icon(node: str) -> str:
    return _ICONS.get(node, "·")


def _col(node: str) -> str:
    return _AGENT_COLOR.get(node, "")


def _model_short(model: str) -> str:
    """Strip provider prefix, return last segment."""
    return model.split("/")[-1] if model else ""


def _mtag(model: str) -> str:
    """Return ' [model-name]' dimmed, or '' if unknown."""
    s = _model_short(model)
    return f" {DIM}[{s}]{RESET}" if s else ""


def show_event(
    event: dict[str, Any],
    detail_level: str = "compact",
    lane_prefix: str = "",
    show_thinking: bool = True,
) -> None:
    et = event.get("type")
    node = event.get("node_name", "node")
    payload = event.get("payload", {})
    lp = f"{DIM}{lane_prefix}{RESET}" if lane_prefix else ""

    if et is None:
        return
    if (node in {"node", "None", "", None} and et == "trace" and not payload) or (not et and not payload):
        return

    if node in {"Planner", "Orchestrator", "Researcher", "Verifier", "Analyst", "Writer"}:
        _STATE.active_agents.add(node)

    icon = _icon(node)
    col = _col(node)

    # ── portfolio events ──────────────────────────────────────────────────────

    if et == "portfolio_lane_event":
        inner = event.get("event", {})
        show_event(inner, detail_level=detail_level, lane_prefix=f"[{event.get('lane_id', '?')}] ", show_thinking=show_thinking)
        return

    if et == "portfolio_plan":
        n = payload.get("lane_count", "?")
        _emit(f"{CYAN}◈ portfolio{RESET}  {n} parallel lanes")
        return

    if et == "portfolio_lane_start":
        if detail_level != "compact":
            _emit(f"  {DIM}├─ lane {payload.get('lane_id')}  {payload.get('provider')}{RESET}")
        return

    if et == "portfolio_lane_done":
        lid = payload.get("lane_id", "?")
        if payload.get("usable"):
            q = payload.get("quality", {})
            _emit(f"  {DIM}└─ lane {lid} ✓  faith={quality_word(q.get('faithfulness'))}{RESET}")
        else:
            err = str(payload.get("error") or "failed")[:60]
            _emit(f"  {YELLOW}└─ lane {lid} ✗  {err}{RESET}")
        return

    if et == "portfolio_summary":
        _emit(f"{GREEN}◈ portfolio{RESET}  selected lane {payload.get('selected')}")
        return

    if et == "orchestration_tree":
        if detail_level != "detail":
            return
        tree = str(payload.get("tree", "")).strip()
        if tree:
            _emit(f"{DIM}{tree}{RESET}")
        return

    # ── agent_message ─────────────────────────────────────────────────────────

    if et == "agent_message":
        if detail_level == "detail":
            src = str(payload.get("from", "")).lower()
            dst = str(payload.get("to", "")).lower()
            _emit(f"  {DIM}{src} → {dst}{RESET}")
        return

    # ── plan_update ───────────────────────────────────────────────────────────

    if et == "plan_update":
        plan = payload.get("plan", [])
        n = len(plan)
        _emit(f"{lp}{col}◆ planner{RESET}  {n} research steps")
        if detail_level in {"brief", "detail"}:
            for i, item in enumerate(plan):
                branch = "└─" if i == n - 1 else "├─"
                words = str(item.get("description", "")).split()
                _emit(f"  {branch} {' '.join(words[:9])}")
        start_spinner("planning")
        return

    # ── reasoning_summary ─────────────────────────────────────────────────────

    if et == "reasoning_summary":
        if not show_thinking:
            return

        # ── Researcher: show actual search query, not boilerplate summary ──
        if node == "Researcher":
            query = str(payload.get("base_query") or payload.get("focus_query") or "").strip()
            route_model = str(payload.get("route_model") or "").strip()
            if detail_level == "compact":
                short_q = query[:80]
                _emit(f'{lp}{col}{icon} researcher{RESET}  "{short_q}"')
            else:
                tag = _mtag(route_model)
                lane = payload.get("lane", "")
                lane_part = f"  {DIM}lane {lane}{RESET}" if lane != "" else ""
                _emit(f'{lp}{col}{icon} researcher{RESET}{tag}  "{query[:100]}"{lane_part}')
            start_spinner("researching")
            return

        # ── Verifier: ALWAYS suppress — the trace event is the display line ──
        if node == "Verifier":
            return

        # ── Analyst: suppress in compact (trace handles it); show in detail ──
        if node == "Analyst":
            if detail_level == "compact":
                return
            route_model = str(payload.get("route_model") or "").strip()
            tag = _mtag(route_model)
            needs_more = payload.get("needs_more_research", False)
            decision = "researching more" if needs_more else "ready to write"
            rationale = str(payload.get("rationale") or "").strip()
            focus = str(payload.get("focus_query") or "").strip()
            parts = []
            if focus:
                parts.append(focus[:80])
            if rationale and detail_level == "detail":
                parts.append(rationale[:160])
            suffix = f"  {DIM}|  {'; '.join(parts)}{RESET}" if parts else ""
            _emit(f"{lp}{col}{icon} analyst{RESET}{tag}  {decision}{suffix}")
            return

        # ── Writer: show source count, not boilerplate ──
        if node == "Writer":
            route_model = str(payload.get("route_model") or "").strip()
            n_sources = payload.get("evidence_used", "?")
            if detail_level == "compact":
                _emit(f"{lp}{col}{icon} writer{RESET}  {n_sources} sources → writing")
            else:
                tag = _mtag(route_model)
                n_chars = payload.get("output_chars", "?")
                _emit(f"{lp}{col}{icon} writer{RESET}{tag}  {n_sources} sources, {n_chars} chars")
            return

        # ── Planner + others: show routing table in detail, brief chip in compact ──
        summary = " ".join(str(payload.get("summary", "")).split())
        routes = payload.get("model_routes")
        route_model = str(payload.get("route_model") or "").strip()
        if detail_level == "compact":
            # For planner, show the routing assignments
            if node == "Planner" and routes:
                assignments = "  ".join(f"{k}={_model_short(v)}" for k, v in routes.items() if v)
                _emit(f"{lp}{col}{icon} planner{RESET}  {DIM}{assignments}{RESET}")
            else:
                tag = _mtag(route_model)
                _emit(f"{lp}{col}{icon} {node.lower()}{RESET}{tag}")
        else:
            tag = _mtag(route_model)
            if node == "Planner" and routes:
                assignments = "  ".join(f"{k}={_model_short(v)}" for k, v in routes.items() if v)
                _emit(f"{lp}{col}{icon} planner{RESET}  {DIM}{assignments}{RESET}")
                rationale = str(payload.get("routing_rationale") or "").strip()
                if rationale and detail_level == "detail":
                    _emit(f"    {DIM}{rationale[:200]}{RESET}")
            else:
                _emit(f"{lp}{col}{icon} {node.lower()}{RESET}{tag}  {summary[:120]}")
        start_spinner(f"{node.lower()}")
        return

    # ── thought_stream ────────────────────────────────────────────────────────

    if et == "thought_stream":
        if not show_thinking:
            return
        summary = " ".join(str(payload.get("summary", "")).split()).strip()
        reason = " ".join(str(payload.get("main_reason", "")).split()).strip()
        if detail_level == "compact":
            if summary:
                _emit(f"{lp}{col}{icon} {node.lower()}{RESET}  {summary[:100]}")
            return
        if summary:
            msg = summary[:120]
            if reason:
                msg = f"{msg}  {DIM}|  {reason[:180]}{RESET}"
            _emit(f"{lp}{col}{icon} {node.lower()}{RESET}  {msg}")
        return

    # ── trace ─────────────────────────────────────────────────────────────────

    if et == "trace":
        # Always suppress noise nodes in compact/brief
        if node in {"PostProcess", "MetricsEmitter", "SaveArtifact"} and detail_level != "detail":
            return

        # ── Researcher: suppress subtask/batch/budget noise ──
        if node == "Researcher":
            if "subtask_id" in payload or "processed_subtasks" in payload or "budget_exhausted" in payload:
                # Only show in detail as raw debug info
                if detail_level == "detail":
                    _emit(f"    {DIM}{json.dumps(payload)[:240]}{RESET}")
                return
            if "search_error" in payload:
                err = payload.get("search_error", {})
                errs = err.get("errors", []) if isinstance(err, dict) else []
                if detail_level == "compact":
                    _emit(f"{lp}{col}{icon} researcher{RESET}  {YELLOW}search failed{RESET}")
                else:
                    first = str(errs[0])[:120] if errs else "unknown"
                    _emit(f"{lp}{col}{icon} researcher{RESET}  {YELLOW}search error: {first}{RESET}")
                return
            if detail_level == "detail":
                _emit(f"    {DIM}{json.dumps(payload)[:240]}{RESET}")
            return

        # ── Verifier: THE display line (trace carries the score) ──
        if node == "Verifier":
            avg = payload.get("avg_verification_score")
            if avg is not None:
                score = round(float(avg), 2)
                contras = payload.get("contradictions", 0)
                route_model = str(payload.get("route_model") or "").strip()
                if detail_level == "compact":
                    _emit(f"{lp}{col}{icon} verifier{RESET}  score {score}")
                else:
                    tag = _mtag(route_model)
                    items = payload.get("items", [])
                    weak = sum(1 for i in items if float(i.get("verification_score", 0)) < 0.6)
                    _emit(f"{lp}{col}{icon} verifier{RESET}{tag}  score {score}  {DIM}|  {weak} weak, {contras} contradictions{RESET}")
            elif detail_level == "detail":
                _emit(f"    {DIM}{json.dumps(payload)[:240]}{RESET}")
            return

        # ── Analyst: show decision in compact; reasoning_summary handles detail ──
        if node == "Analyst":
            if "needs_more_research" in payload:
                if detail_level == "compact":
                    verb = "researching more" if payload.get("needs_more_research") else "ready to write"
                    _emit(f"{lp}{col}{icon} analyst{RESET}  {verb}")
                # In detail mode, the reasoning_summary line has the full rationale
                return
            if detail_level == "detail":
                _emit(f"    {DIM}{json.dumps(payload)[:240]}{RESET}")
            return

        # ── Writer: suppress in compact ──
        if node == "Writer":
            if detail_level != "detail":
                return
            if "warning" in payload:
                _emit(f"{lp}{col}{icon} writer{RESET}  {YELLOW}⚠ {payload['warning'][:120]}{RESET}")
            else:
                _emit(f"    {DIM}{json.dumps(payload)[:240]}{RESET}")
            return

        # ── Other nodes (Planner, HITL, etc.) ──
        if detail_level == "compact":
            return
        if detail_level == "brief":
            key_bits = [
                f"{k}={payload[k]}" for k in
                ["subtask_id", "cycle_count", "needs_more_research",
                 "context_precision", "citation_count", "avg_verification_score"]
                if k in payload
            ]
            body = ", ".join(key_bits)[:200] if key_bits else json.dumps(payload)[:160]
        else:
            body = json.dumps(payload)[:400]
        _emit(f"{col}{icon} {node.lower()}{RESET}  {DIM}{body}{RESET}")
        return

    # ── evidence_item ─────────────────────────────────────────────────────────

    if et == "evidence_item":
        _STATE.evidence_count += 1
        title = str(payload.get("title") or payload.get("source_url", "source"))
        short = " ".join(title.split()[:7])
        _emit(f"{lp}{GREEN}◈ found{RESET}  {short}")
        start_spinner("fetching")
        return

    # ── markdown outputs — suppressed during stream; printed after completion ─

    if et in {"draft_markdown", "final_markdown"}:
        return

    # ── hitl ──────────────────────────────────────────────────────────────────

    if et == "hitl_request":
        q = str(payload.get("question", "Input required"))
        _STATE.pending_hitl = q
        _STATE.session_state = "waiting_hitl"
        _emit(f"\n{YELLOW}? {q[:180]}{RESET}")
        return

    # ── done ──────────────────────────────────────────────────────────────────

    if et == "done":
        _STATE.pending_hitl = ""
        _STATE.session_state = str(payload.get("status", "done"))
        elapsed = time.time() - _SESSION_START
        _emit(f"\n{GREEN}✓ done{RESET}  {_STATE.evidence_count} sources  {elapsed:.1f}s")
        return

    # ── fallback ──────────────────────────────────────────────────────────────

    if detail_level != "compact":
        _emit(f"{DIM}{et} {lane_prefix}{node.lower()}: {json.dumps(payload)[:180]}{RESET}")


atexit.register(shutdown_tui)
