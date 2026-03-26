from __future__ import annotations

import json
import shlex
import subprocess
import sys
import base64
import os
from pathlib import Path
from typing import Any, Callable

try:  # pragma: no cover
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.shortcuts import CompleteStyle
except Exception:  # pragma: no cover
    PromptSession = None  # type: ignore[assignment]
    WordCompleter = None  # type: ignore[assignment]
    CompleteStyle = None  # type: ignore[assignment]
try:  # pragma: no cover
    from rich.console import Console
except Exception:  # pragma: no cover
    Console = None  # type: ignore[assignment]


CLI_COMMANDS_PATH = Path.home() / ".research_swarm" / "commands.json"
DEFAULT_COMMANDS = [
    {"name": "/help", "description": "Show command menu and usage", "icon": "help.svg"},
    {"name": "/commands", "description": "List all slash commands", "icon": "terminal.svg"},
    {"name": "/sessions", "description": "List active + persisted sessions", "icon": "history.svg"},
    {"name": "/resume", "description": "Resume a session id: /resume <id>", "icon": "play.svg"},
    {"name": "/setup", "description": "Run provider key setup wizard", "icon": "settings.svg"},
    {"name": "/mode", "description": "Set run mode: /mode single|portfolio", "icon": "layers.svg"},
    {"name": "/budget", "description": "Set budget: /budget low|balanced|high", "icon": "coins.svg"},
    {"name": "/depth", "description": "Set depth: /depth quick|standard|deep", "icon": "brain.svg"},
    {"name": "/provider", "description": "Set provider preference", "icon": "network.svg"},
    {"name": "/memory", "description": "Cross-session memory: /memory on|off", "icon": "memory.svg"},
    {"name": "/view", "description": "Set stream view: /view brief|detail", "icon": "eye.svg"},
    {"name": "/compact", "description": "Set stream view compact", "icon": "eye.svg"},
    {"name": "/detail", "description": "Set stream view detail", "icon": "eye.svg"},
    {"name": "/thinking", "description": "Thinking summaries: /thinking on|off", "icon": "spark.svg"},
    {"name": "/playwright", "description": "Install/check Playwright browser for deep search", "icon": "terminal.svg"},
]
ICONS_DIR = Path(__file__).resolve().parent / "icons" / "commands"
NERD_ICON_MAP = {
    "help.svg": "\uf059",
    "terminal.svg": "\uf489",
    "history.svg": "\uf1da",
    "play.svg": "\uf04b",
    "settings.svg": "\uf013",
    "layers.svg": "\uf5fd",
    "coins.svg": "\uf51e",
    "brain.svg": "\uf5dc",
    "network.svg": "\uf6ff",
    "memory.svg": "\uf538",
    "eye.svg": "\uf06e",
    "spark.svg": "\uf890",
}


def _supports_inline_svg() -> bool:
    return os.getenv("TERM_PROGRAM") in {"iTerm.app", "WezTerm"}


def _inline_svg(icon_name: str) -> str:
    p = ICONS_DIR / icon_name
    if not p.exists():
        return ""
    encoded = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"\033]1337;File=name={icon_name};inline=1;width=1;height=1;preserveAspectRatio=1:{encoded}\a"


def load_command_specs() -> list[dict[str, str]]:
    specs = list(DEFAULT_COMMANDS)
    if not CLI_COMMANDS_PATH.exists():
        return specs
    try:
        payload = json.loads(CLI_COMMANDS_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name", "")).strip()
                if not name.startswith("/"):
                    continue
                desc = str(row.get("description", "")).strip() or "Custom command"
                icon = str(row.get("icon", "custom.svg")).strip() or "custom.svg"
                specs.append({"name": name, "description": desc, "icon": icon})
    except Exception:
        return specs
    return specs


def show_command_menu() -> None:
    rows = load_command_specs()
    if Console is not None:
        c = Console()
        c.print("[bold cyan]Slash Commands[/]")
        for row in rows:
            name = row.get("name", "")
            desc = row.get("description", "")
            icon = row.get("icon", "")
            if _supports_inline_svg():
                inline = _inline_svg(icon)
                icon_part = inline if inline else NERD_ICON_MAP.get(icon, "\uf128")
                print(f"{icon_part} {name} :: {desc}")
            else:
                glyph = NERD_ICON_MAP.get(icon, "\uf128")
                c.print(f"[bold magenta on black]{glyph}[/] [bright_cyan]{name}[/] :: {desc}")
        return
    print("Slash Commands")
    for row in rows:
        icon = row.get("icon", "")
        glyph = NERD_ICON_MAP.get(icon, "*")
        print(f"- {glyph} {row.get('name','')} :: {row.get('description','')}")


def smart_prompt(prompt: str, default: str = "") -> str:
    if not sys.stdin.isatty():
        return input(prompt)
    if PromptSession is None or WordCompleter is None or CompleteStyle is None:
        return input(prompt)
    words = [row.get("name", "") for row in load_command_specs() if row.get("name")]
    completer = WordCompleter(words, ignore_case=True, sentence=True, match_middle=True)
    session = PromptSession()
    return session.prompt(
        prompt,
        default=default,
        completer=completer,
        complete_while_typing=True,
        complete_style=CompleteStyle.MULTI_COLUMN,
        reserve_space_for_menu=8,
    )


def handle_slash_command(
    raw: str,
    *,
    api_base: str,
    args: Any,
    body: dict[str, Any] | None,
    list_sessions_cb: Callable[[str], list[dict[str, str]]],
    provider_setup_cb: Callable[[], None],
) -> dict[str, Any]:
    line = raw.strip()
    if not line.startswith("/"):
        return {"handled": False}
    try:
        parts = shlex.split(line)
    except Exception:
        parts = line.split()
    cmd = parts[0].lower()
    rest = parts[1:]

    if cmd in {"/help", "/commands"}:
        show_command_menu()
        return {"handled": True}
    if cmd == "/sessions":
        sessions = list_sessions_cb(api_base)
        if not sessions:
            print("No sessions found.")
        else:
            print("Sessions:")
            for row in sessions:
                src = row.get("source", "runtime")
                print(f"- {row.get('session_id')} [{row.get('status')}] source={src}")
        return {"handled": True}
    if cmd == "/resume":
        if not rest:
            print("Usage: /resume <session_id>")
            return {"handled": True}
        return {"handled": True, "action": "resume", "session_id": rest[0]}
    if cmd == "/setup":
        provider_setup_cb()
        return {"handled": True}
    if cmd == "/mode" and body is not None:
        if rest and rest[0] in {"single", "portfolio"}:
            body["mode"] = rest[0]
            print(f"mode={body['mode']}")
        else:
            print("Usage: /mode single|portfolio")
        return {"handled": True}
    if cmd == "/budget" and body is not None:
        if rest and rest[0] in {"low", "balanced", "high"}:
            body["budget_mode"] = rest[0]
            print(f"budget_mode={body['budget_mode']}")
        else:
            print("Usage: /budget low|balanced|high")
        return {"handled": True}
    if cmd == "/depth" and body is not None:
        if rest and rest[0] in {"quick", "standard", "deep"}:
            body["depth"] = rest[0]
            print(f"depth={body['depth']}")
        else:
            print("Usage: /depth quick|standard|deep")
        return {"handled": True}
    if cmd == "/provider" and body is not None:
        allowed = {"auto", "mixed", "groq", "together", "gemini", "openai", "anthropic"}
        if rest and rest[0] in allowed:
            body["provider_pref"] = rest[0]
            print(f"provider_pref={body['provider_pref']}")
        else:
            print("Usage: /provider auto|mixed|groq|together|gemini|openai|anthropic")
        return {"handled": True}
    if cmd == "/memory" and body is not None:
        if rest and rest[0] in {"on", "off"}:
            body["cross_session_memory"] = rest[0] == "on"
            print(f"cross_session_memory={body['cross_session_memory']}")
        else:
            print("Usage: /memory on|off")
        return {"handled": True}
    if cmd == "/view":
        if rest and rest[0] in {"brief", "detail", "compact"}:
            args.detail_level = rest[0]
            if body is not None:
                body["detail_level"] = rest[0]
            print(f"detail_level={args.detail_level}")
        else:
            print("Usage: /view brief|detail|compact")
        return {"handled": True}
    if cmd == "/compact":
        args.detail_level = "compact"
        if body is not None:
            body["detail_level"] = "compact"
        print("detail_level=compact")
        return {"handled": True}
    if cmd == "/detail":
        args.detail_level = "detail"
        if body is not None:
            body["detail_level"] = "detail"
        print("detail_level=detail")
        return {"handled": True}
    if cmd == "/thinking":
        if rest and rest[0] in {"on", "off"}:
            args.thinking_default = rest[0]
            print(f"thinking={args.thinking_default}")
        else:
            print("Usage: /thinking on|off")
        return {"handled": True}
    if cmd == "/playwright":
        # Check current status
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                browser.close()
            print("\033[92m✓ Playwright is installed and working.\033[0m")
            print("  Chromium browser ready. Playwright will be used as the primary search provider.")
            return {"handled": True}
        except ImportError:
            print("\033[93mPlaywright not installed.\033[0m")
        except Exception as exc:
            print(f"\033[93mPlaywright installed but chromium missing: {exc}\033[0m")

        answer = input("Install Playwright + Chromium now? (y/N): ").strip().lower()
        if answer in {"y", "yes"}:
            print("Installing playwright package...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-U", "playwright"], check=False)
            print("Installing Chromium browser...")
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=False)
            # Verify
            try:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    browser.close()
                # Clear cached availability check
                from tools.mcp_tools import playwright_available
                playwright_available.cache_clear()
                print("\033[92m✓ Playwright installed successfully! It will now be used as the primary search provider.\033[0m")
            except Exception as exc:
                print(f"\033[91m✗ Installation issue: {exc}\033[0m")
        return {"handled": True}
    print(f"Unknown command: {cmd}. Use /commands")
    return {"handled": True}
