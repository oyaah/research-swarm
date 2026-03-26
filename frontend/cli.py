from __future__ import annotations

import argparse
import json
import os
import contextlib
import re
import select
import subprocess
import termios
import tty
import sys
import time
import urllib.parse
from typing import Any

import requests

from frontend.cli_commands import handle_slash_command, smart_prompt
from frontend.cli_events import (
    BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW,
    get_ui as _get_ui,
    render_cyberpunk_banner as _render_cyberpunk_banner,
    show_event as _show_event,
    start_spinner as _start_spinner,
    stop_spinner as _stop_spinner,
    wrap as _wrap,
)
from frontend.cli_http import fetch_status as _fetch_status
from frontend.cli_http import list_sessions_http as _list_sessions
from frontend.cli_http import parse_sse_lines as _parse_sse_lines
from frontend.cli_http import resume_session_http as _resume
from frontend.cli_http import start_session as _start_session
from frontend.cli_http import steer_session_http as _steer
from frontend.cli_setup import (
    CLI_SETUP_PATH,
    interactive_provider_setup,
    load_cli_setup,
    run_first_time_setup,
)


_OBSIDIAN_APPS = ["/Applications/Obsidian.app", os.path.expanduser("~/Applications/Obsidian.app")]


def _open_in_obsidian(canvas_path: str) -> bool:
    """Try to open a .canvas file in Obsidian. Returns True if launched."""
    if not any(os.path.isdir(p) for p in _OBSIDIAN_APPS):
        return False
    abs_path = os.path.abspath(canvas_path)
    uri = f"obsidian://open?path={urllib.parse.quote(abs_path)}"
    try:
        subprocess.Popen(["open", uri])
        return True
    except Exception:
        return False


def _show_artifacts(artifacts: dict[str, Any], session_id: str) -> None:
    if artifacts:
        print(f"\n{BOLD}{CYAN}Generated Artifacts:{RESET}")
        for k, v in artifacts.items():
            print(f"  - {k}: {v}")
        canvas = artifacts.get("knowledge_map") or artifacts.get("canvas", "")
        if canvas and os.path.exists(canvas):
            if _open_in_obsidian(canvas):
                print(f"{DIM}  → opened in Obsidian{RESET}")
            else:
                print(f"{DIM}  → open {canvas} in Obsidian to explore the knowledge map{RESET}")
    else:
        print(f"\n{DIM}Full report path: artifacts/reports/{session_id}/report.md{RESET}")


def _print_final_report(markdown: str) -> None:
    if not markdown.strip():
        return
    _stop_spinner()
    divider = "═" * 72
    print(f"\n{BOLD}{CYAN}{divider}{RESET}")
    print(f"{BOLD}{CYAN}  RESEARCH REPORT{RESET}")
    print(f"{BOLD}{CYAN}{divider}{RESET}\n")
    for line in markdown.splitlines():
        if line.startswith("# "):
            print(f"{BOLD}{line}{RESET}")
        elif line.startswith("## "):
            print(f"{BOLD}{CYAN}{line}{RESET}")
        elif line.startswith("### "):
            print(f"{CYAN}{line}{RESET}")
        else:
            print(line)
    print(f"\n{DIM}{'─' * 72}{RESET}\n")


@contextlib.contextmanager
def _cbreak_mode():
    """Single-key input mode during streaming. Preserves Ctrl+C."""
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _api_preflight(api_base: str) -> tuple[bool, str]:
    try:
        res = requests.get(f"{api_base}/v1/tools", timeout=3)
        if 200 <= res.status_code < 500:
            return True, ""
        return False, f"unexpected status {res.status_code}"
    except Exception as exc:
        return False, str(exc)


def _explain_start_error(exc: Exception, api_base: str) -> None:
    _stop_spinner()
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        detail = ""
        try:
            payload = exc.response.json()
            detail = str(payload.get("detail", payload))
        except Exception:
            detail = exc.response.text[:400]
        print(f"{RED}✗ HTTP {status}{RESET}" + (f"  {detail}" if detail else ""))
        return
    if isinstance(exc, (requests.ConnectionError, requests.ReadTimeout, requests.Timeout)):
        print(f"{RED}✗ API unreachable at {api_base}{RESET}")
        print("  hint: uvicorn api.app:app --host 0.0.0.0 --port 8000")
        return
    print(f"{RED}✗ {exc}{RESET}")


def _stream_once(
    api_base: str,
    session_id: str,
    detail_level_ref: list[str],
    thinking_ref: list[bool],
    args: argparse.Namespace,
) -> str:
    _start_spinner("planning")
    last_exc: Exception | None = None
    pending_hitl: str = ""
    pending_hitl_counter = 0
    pending_hitl_last_print = 0.0
    pending_hitl_target_session = session_id
    _input_buf: list[str] = []

    for attempt in range(8):
        try:
            with _cbreak_mode(), requests.get(f"{api_base}/v1/stream/{session_id}", stream=True, timeout=300) as resp:
                resp.raise_for_status()
                done_status = "failed"
                for _, payload in _parse_sse_lines(resp):
                    try:
                        event = json.loads(payload)
                    except Exception:
                        continue
                    _show_event(event, detail_level=detail_level_ref[0], show_thinking=thinking_ref[0])

                    et = event.get("type")
                    if et == "hitl_request":
                        pending_hitl = str(event.get("payload", {}).get("question", "")).strip()
                        pending_hitl_target_session = session_id
                        pending_hitl_counter = 0
                    elif et == "portfolio_lane_event":
                        inner = event.get("event", {})
                        if inner.get("type") == "hitl_request":
                            lane = event.get("lane_id", "")
                            q = str(inner.get("payload", {}).get("question", "")).strip()
                            pending_hitl = f"[{lane}] {q}" if lane else q
                            pending_hitl_target_session = str(event.get("child_session_id", session_id))
                            pending_hitl_counter = 0

                    if pending_hitl and et != "hitl_request":
                        pending_hitl_counter += 1
                        now = time.time()
                        if pending_hitl_counter % 3 == 0 and (now - pending_hitl_last_print) >= 2.0:
                            _stop_spinner()
                            print(f"{YELLOW}? {_wrap(pending_hitl, 110)}{RESET}")
                            pending_hitl_last_print = now

                    if sys.stdin.isatty():
                        try:
                            ready, _, _ = select.select([sys.stdin], [], [], 0)
                        except Exception:
                            ready = []
                        if ready:
                            raw = os.read(sys.stdin.fileno(), 256)
                            for byte_val in raw:
                                ch = chr(byte_val)
                                if byte_val == 0x0f:
                                    detail_level_ref[0] = "detail"
                                    _stop_spinner()
                                    sys.stdout.write(f"\r{DIM}→ detail mode{RESET}\n")
                                    sys.stdout.flush()
                                    _start_spinner("running")
                                    _input_buf.clear()
                                elif byte_val == 0x02:
                                    detail_level_ref[0] = "compact"
                                    _stop_spinner()
                                    sys.stdout.write(f"\r{DIM}→ compact mode{RESET}\n")
                                    sys.stdout.flush()
                                    _start_spinner("running")
                                    _input_buf.clear()
                                elif ch in ("\r", "\n"):
                                    sys.stdout.write("\r\n")
                                    sys.stdout.flush()
                                    line = "".join(_input_buf).strip()
                                    _input_buf.clear()
                                    if line:
                                        low = line.lower()
                                        if pending_hitl:
                                            answer = line or "approve"
                                            _resume(api_base, pending_hitl_target_session, answer)
                                            _stop_spinner()
                                            print(f"{YELLOW}→ sent: {_wrap(answer, 90)}{RESET}")
                                            _get_ui().pending_hitl = ""
                                            _get_ui().session_state = "running"
                                            pending_hitl = ""
                                            pending_hitl_counter = 0
                                            pending_hitl_last_print = 0.0
                                            _start_spinner("resuming")
                                        elif low.startswith("/"):
                                            _ = handle_slash_command(
                                                low, api_base=api_base, args=args, body=None,
                                                list_sessions_cb=_list_sessions,
                                                provider_setup_cb=interactive_provider_setup,
                                            )
                                        elif low.startswith("steer:"):
                                            steer_msg = line.split(":", 1)[1].strip()
                                            if steer_msg:
                                                _steer(api_base, session_id, steer_msg)
                                                _stop_spinner()
                                                print(f"{YELLOW}→ steering: {steer_msg}{RESET}")
                                                _start_spinner("steering")
                                        elif low in {"detail", "o"}:
                                            detail_level_ref[0] = "detail"
                                            _stop_spinner()
                                            print(f"{DIM}→ detail mode{RESET}")
                                            _start_spinner("running")
                                        elif low in {"compact", "d", "b"}:
                                            detail_level_ref[0] = "compact"
                                            _stop_spinner()
                                            print(f"{DIM}→ compact mode{RESET}")
                                            _start_spinner("running")
                                        elif low == "brief":
                                            detail_level_ref[0] = "brief"
                                            _stop_spinner()
                                            print(f"{DIM}→ brief mode{RESET}")
                                            _start_spinner("running")
                                        elif low in {"thinking on", "t on"}:
                                            thinking_ref[0] = True
                                        elif low in {"thinking off", "t off"}:
                                            thinking_ref[0] = False
                                elif byte_val in (0x7f, 0x08):
                                    if _input_buf:
                                        _input_buf.pop()
                                        sys.stdout.write("\b \b")
                                        sys.stdout.flush()
                                elif ch.isprintable():
                                    _input_buf.append(ch)
                                    sys.stdout.write(ch)
                                    sys.stdout.flush()

                    if et == "done":
                        done_status = event.get("payload", {}).get("status", "failed")

                _stop_spinner()
                if done_status == "waiting_hitl" and pending_hitl:
                    print(f"{YELLOW}? {_wrap(pending_hitl, 110)}{RESET}")
                return done_status

        except requests.HTTPError as exc:
            _stop_spinner()
            if exc.response is not None and exc.response.status_code == 404:
                try:
                    sp = _fetch_status(api_base, session_id)
                    st = str(sp.get("status", "failed"))
                    if st in {"completed", "failed", "waiting_hitl"}:
                        return st
                except Exception:
                    pass
                if attempt < 7:
                    time.sleep(0.25 * (attempt + 1))
                    last_exc = exc
                    continue
            raise
        except requests.RequestException as exc:
            _stop_spinner()
            last_exc = exc
            if attempt < 7:
                time.sleep(0.25 * (attempt + 1))
                _start_spinner("reconnecting")
                continue
            raise

    if last_exc:
        raise last_exc
    return "failed"


def _run_hitl_loop(
    api_base: str,
    session_id: str,
    detail_level_ref: list[str],
    thinking_ref: list[bool],
    auto_approve: bool,
    args: argparse.Namespace,
    default_prompt: str = "Approve or request changes:",
) -> str:
    """Stream + handle HITL gates until done/failed. Returns final stream status."""
    while True:
        st = _stream_once(api_base, session_id, detail_level_ref=detail_level_ref,
                          thinking_ref=thinking_ref, args=args)
        if st != "waiting_hitl":
            return st
        current = _fetch_status(api_base, session_id)
        prompt = current.get("final_state", {}).get("hitl_question", default_prompt)
        if auto_approve:
            answer = "approve"
            print(f"{YELLOW}→ auto-approve{RESET}")
        else:
            print(f"{YELLOW}? {prompt}{RESET}")
            answer = input(f"{DIM}> {RESET}").strip() or "approve"
        _resume(api_base, session_id, answer)
        _get_ui().pending_hitl = ""
        _get_ui().session_state = "running"
        _start_spinner("resuming")


def _finalize_session(api_base: str, session_id: str, missing_err: str = "unknown error") -> int:
    """Fetch final state, print report and artifacts. Returns exit code."""
    final = _fetch_status(api_base, session_id)
    if final.get("status") != "completed":
        err = final.get("final_state", {}).get("error", missing_err)
        print(f"{RED}✗ {err}{RESET}")
        return 1
    md = final.get("final_state", {}).get("final_markdown", "")
    _print_final_report(md)
    _show_artifacts(final.get("final_state", {}).get("artifacts", {}), session_id)
    return 0


def run_cli(api_base: str, body: dict[str, Any], auto_approve: bool, args: argparse.Namespace) -> int:
    _start_spinner("starting")
    try:
        session_id = _start_session(api_base, body)
    except Exception as exc:
        _explain_start_error(exc, api_base)
        return 2

    _stop_spinner()
    print(f"{DIM}session {session_id}{RESET}")

    detail_level_ref = [body.get("detail_level", "compact")]
    thinking_ref = [body.get("thinking_default", "on") != "off"]
    _run_hitl_loop(api_base, session_id, detail_level_ref, thinking_ref, auto_approve, args)
    return _finalize_session(api_base, session_id)


def resume_cli(api_base: str, session_id: str, auto_approve: bool, detail_level: str, args: argparse.Namespace) -> int:
    print(f"{DIM}resuming {session_id}{RESET}")
    try:
        current = _fetch_status(api_base, session_id)
    except Exception as exc:
        _explain_start_error(exc, api_base)
        return 2
    if current.get("status") in {"completed", "failed"}:
        err = current.get("final_state", {}).get("error", "")
        print(f"{RED}✗ already ended: {err or current.get('status')}{RESET}")
        return 1

    detail_level_ref = [detail_level]
    thinking_ref = [True]
    _run_hitl_loop(api_base, session_id, detail_level_ref, thinking_ref, auto_approve, args,
                   default_prompt="Approve:")
    return _finalize_session(api_base, session_id, missing_err="failed")


def _build_body(query: str, args: argparse.Namespace) -> dict[str, Any]:
    normalized_query = re.sub(r"^\s*(?:>\s*)+", "", query or "").strip()
    return {
        "query": normalized_query,
        "mode": args.mode,
        "budget_mode": args.budget_mode,
        "depth": args.depth,
        "provider_pref": args.provider_pref,
        "lane_preference": args.lane_preference,
        "detail_level": args.detail_level,
        "preference_text": args.preference_text,
        "cross_session_memory": args.cross_session_memory,
        "thinking_default": args.thinking_default,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Research Swarm")
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--query", default="")
    parser.add_argument("--auto-approve", action="store_true")
    parser.add_argument("--mode", choices=["single", "portfolio"], default="single")
    parser.add_argument("--budget-mode", choices=["low", "balanced", "high"], default="balanced")
    parser.add_argument("--depth", choices=["quick", "standard", "deep"], default="standard")
    parser.add_argument("--provider-pref", choices=["auto", "mixed", "groq", "together", "gemini", "openai", "anthropic"], default="auto")
    parser.add_argument("--lane-preference", choices=["auto", "both", "fast", "deep"], default="auto")
    parser.add_argument("--detail-level", choices=["compact", "brief", "detail"], default="compact")
    parser.add_argument("--preference-text", default="")
    parser.add_argument("--cross-session-memory", action="store_true")
    parser.add_argument("--resume-session", default="")
    parser.add_argument("--thinking-default", choices=["on", "off"], default="on")
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--reset-setup", action="store_true")
    args = parser.parse_args()

    ok, why = _api_preflight(args.api_base)
    if not ok:
        print(f"{RED}✗ API unreachable: {args.api_base}{RESET}")
        if why:
            print(f"  {why}")
        print("  hint: uvicorn api.app:app --host 0.0.0.0 --port 8000")
        return 2

    if args.reset_setup and CLI_SETUP_PATH.exists():
        CLI_SETUP_PATH.unlink()
    setup_cfg = load_cli_setup()
    if sys.stdin.isatty() and (args.setup or not setup_cfg.get("initialized", False)):
        defaults = {
            "mode": args.mode, "budget_mode": args.budget_mode,
            "depth": args.depth, "provider_pref": args.provider_pref,
            "detail_level": args.detail_level,
        }

        def _menu_select(title: str, options: list[str], default_idx: int = 0) -> int:
            print(f"\n{title}")
            for i, opt in enumerate(options):
                print(f"  {'>' if i == default_idx else ' '} {i+1}. {opt}")
            raw = input("> ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return int(raw) - 1
            return default_idx

        setup_cfg = run_first_time_setup(defaults, _menu_select)

    if setup_cfg.get("initialized", False):
        if args.mode == "single":
            args.mode = setup_cfg.get("mode", args.mode)
        if args.budget_mode == "balanced":
            args.budget_mode = setup_cfg.get("budget_mode", args.budget_mode)
        if args.depth == "standard":
            args.depth = setup_cfg.get("depth", args.depth)
        if args.provider_pref == "auto":
            args.provider_pref = setup_cfg.get("provider_pref", args.provider_pref)
        if args.detail_level == "compact":
            args.detail_level = setup_cfg.get("detail_level", args.detail_level)
        if not args.preference_text.strip():
            args.preference_text = setup_cfg.get("preference_text", args.preference_text)
        if not args.auto_approve:
            args.auto_approve = bool(setup_cfg.get("auto_approve_default", False))
        if not args.cross_session_memory:
            args.cross_session_memory = bool(setup_cfg.get("cross_session_memory_default", False))

    _render_cyberpunk_banner()
    print(f"{DIM}Ctrl+O detail  Ctrl+B compact  (or type o/d + Enter)  steer: <msg>{RESET}")
    print()  # breathing room after banner

    if args.resume_session.strip():
        return resume_cli(args.api_base, args.resume_session.strip(),
                          auto_approve=args.auto_approve, detail_level=args.detail_level, args=args)

    if args.query.strip():
        body = _build_body(args.query.strip(), args)
        return run_cli(args.api_base, body, auto_approve=args.auto_approve, args=args)

    if not sys.stdin.isatty():
        print(f"{RED}✗ provide --query in non-interactive mode{RESET}")
        return 2

    first = True
    while True:
        if not first:
            print(f"\n{DIM}{'─' * 48}{RESET}\n")
        first = False

        lane_text = f"  ·  lane: {args.lane_preference}" if args.mode == "portfolio" else ""
        settings_line = f"mode: {args.mode}  ·  budget: {args.budget_mode}  ·  depth: {args.depth}  ·  provider: {args.provider_pref}{lane_text}"
        print(f"{DIM}  {settings_line}   (override: ::budget=high ::mode=portfolio ::depth=deep){RESET}")
        print(f"{BOLD}What do you want to research?{RESET}")
        if args.mode == "portfolio" and args.provider_pref in {"auto", "mixed", "groq"}:
            print(f"{DIM}Portfolio lane: [1] both  [2] groq_fast  [3] groq_deep  (Enter keeps current: {args.lane_preference}){RESET}")
            lane_choice = smart_prompt("> ").strip().lower()
            m = re.search(r"([123])", lane_choice)
            if m:
                lane_choice = m.group(1)
            if lane_choice in {"1", "both"}:
                args.lane_preference = "both"
            elif lane_choice in {"2", "fast", "groq_fast"}:
                args.lane_preference = "fast"
            elif lane_choice in {"3", "deep", "groq_deep"}:
                args.lane_preference = "deep"
        while True:
            query = smart_prompt("> ").strip()
            if not query:
                continue
            cmd = handle_slash_command(
                query, api_base=args.api_base, args=args, body=None,
                list_sessions_cb=_list_sessions,
                provider_setup_cb=interactive_provider_setup,
            )
            if cmd.get("handled"):
                if cmd.get("action") == "resume":
                    resume_cli(args.api_base, cmd.get("session_id", ""),
                               auto_approve=args.auto_approve, detail_level=args.detail_level, args=args)
                    break
                if cmd.get("action") == "quit":
                    return 0
                continue
            break

        if not query:
            continue

        # Parse inline options: "query ::budget=high ::depth=deep ::provider=groq"
        defaults = _build_body("", args)
        body = _build_body(query, args)
        if "::" in query:
            parts = query.split("::")
            body["query"] = parts[0].strip()
            for part in parts[1:]:
                kv = part.strip().split("=", 1)
                if len(kv) == 2:
                    k, v = kv[0].strip().lower(), kv[1].strip().lower()
                    if k == "budget" and v in {"low", "balanced", "high"}:
                        body["budget_mode"] = v
                    elif k == "depth" and v in {"quick", "standard", "deep"}:
                        body["depth"] = v
                    elif k == "provider" and v in {"auto", "mixed", "groq", "together", "gemini", "openai", "anthropic"}:
                        body["provider_pref"] = v
                    elif k == "lane" and v in {"auto", "both", "fast", "deep", "groq_fast", "groq_deep"}:
                        body["lane_preference"] = {"groq_fast": "fast", "groq_deep": "deep"}.get(v, v)
                    elif k == "mode" and v in {"single", "portfolio"}:
                        body["mode"] = v
            if body["query"] != query:
                overrides = {k: body[k] for k in ("budget_mode", "depth", "provider_pref", "lane_preference", "mode") if body.get(k) != defaults.get(k)}
                if overrides:
                    print(f"{DIM}  overrides: {overrides}{RESET}")

        run_cli(args.api_base, body, auto_approve=args.auto_approve, args=args)


if __name__ == "__main__":
    raise SystemExit(main())
