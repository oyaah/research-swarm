from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from dotenv import dotenv_values


CLI_SETUP_PATH = Path.home() / ".research_swarm" / "cli_setup.json"


def provider_ready_from_env(provider: str) -> bool:
    env = dotenv_values(".env") if os.path.exists(".env") else {}
    val = os.environ.get
    if provider == "groq":
        return bool((val("GROQ_API_KEY") or env.get("GROQ_API_KEY") or "").strip())
    if provider == "together":
        return bool((val("DEEPSEEK_API_KEY") or env.get("DEEPSEEK_API_KEY") or "").strip())
    if provider == "gemini":
        return bool((val("GEMINI_API_KEY") or env.get("GEMINI_API_KEY") or "").strip())
    if provider == "openai":
        return bool((val("OPENAI_API_KEY") or env.get("OPENAI_API_KEY") or "").strip())
    if provider == "anthropic":
        return bool((val("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_API_KEY") or "").strip())
    return False


def provider_order_preview(provider_pref: str, budget_mode: str, preference_text: str) -> list[str]:
    pref_text = preference_text.lower()
    premium = budget_mode == "high" or any(
        k in pref_text for k in ["no budget", "no problem with money", "best quality", "premium"]
    )
    if provider_pref in {"groq", "together", "gemini", "openai", "anthropic"}:
        ordered = [provider_pref]
    elif provider_pref == "mixed":
        ordered = ["anthropic", "openai", "groq", "together", "gemini"] if premium else ["groq", "anthropic", "openai", "together", "gemini"]
    else:
        if premium:
            ordered = ["anthropic", "openai", "groq", "together", "gemini"]
        elif budget_mode == "low":
            ordered = ["groq", "gemini", "together", "openai", "anthropic"]
        else:
            ordered = ["groq", "anthropic", "openai", "together", "gemini"]
    return [p for p in ordered if provider_ready_from_env(p)]


def save_env_key(key: str, value: str, env_path: str = ".env") -> None:
    if not value.strip():
        return
    current = dotenv_values(env_path) if os.path.exists(env_path) else {}
    current[key] = value.strip()
    lines = [f"{k}={v}" for k, v in current.items() if v is not None]
    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def load_cli_setup() -> dict[str, Any]:
    if not CLI_SETUP_PATH.exists():
        return {}
    try:
        return json.loads(CLI_SETUP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cli_setup(data: dict[str, Any]) -> None:
    CLI_SETUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLI_SETUP_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def interactive_provider_setup() -> None:
    print("\nProvider setup (leave blank to skip):")
    mapping = [
        ("GROQ_API_KEY", "Groq API key"),
        ("DEEPSEEK_API_KEY", "Together/DeepSeek API key"),
        ("GEMINI_API_KEY", "Gemini API key"),
        ("OPENAI_API_KEY", "OpenAI API key"),
        ("ANTHROPIC_API_KEY", "Anthropic API key"),
    ]
    existing_env = dotenv_values(".env") if os.path.exists(".env") else {}
    for env_name, label in mapping:
        existing = os.environ.get(env_name, "") or str(existing_env.get(env_name, "") or "")
        if existing:
            print(f"- {label}: already set")
            continue
        val = input(f"- {label}: ").strip()
        if val:
            os.environ[env_name] = val
            save_env_key(env_name, val)
    print("Provider setup saved.\n")


def run_first_time_setup(defaults: dict[str, Any], menu_select: Callable[[str, list[str], int], int]) -> dict[str, Any]:
    print(f"First-time setup (saved at {CLI_SETUP_PATH})")
    mode_opts = ["single", "portfolio"]
    budget_opts = ["low", "balanced", "high"]
    depth_opts = ["quick", "standard", "deep"]
    provider_opts = ["auto", "mixed", "groq", "together", "gemini", "openai", "anthropic"]
    detail_opts = ["compact", "brief", "detail"]

    mode = mode_opts[menu_select("Default run mode", mode_opts, mode_opts.index(defaults.get("mode", "portfolio")))]
    budget = budget_opts[menu_select("Default budget mode", budget_opts, budget_opts.index(defaults.get("budget_mode", "balanced")))]
    depth = depth_opts[menu_select("Default research depth", depth_opts, depth_opts.index(defaults.get("depth", "standard")))]
    provider = provider_opts[menu_select("Default provider preference", provider_opts, provider_opts.index(defaults.get("provider_pref", "auto")))]
    default_detail = defaults.get("detail_level", "brief")
    if default_detail not in detail_opts:
        default_detail = "brief"
    detail = detail_opts[menu_select("Default live detail view", detail_opts, detail_opts.index(default_detail))]

    auto_approve_default = input("Default auto-approve HITL? (y/N): ").strip().lower() in {"y", "yes"}
    cross_session_memory_default = input("Default cross-session memory recall? (y/N): ").strip().lower() in {"y", "yes"}
    preference_text = input("Default preference text (optional):\n> ").strip()
    setup = {
        "initialized": True,
        "mode": mode,
        "budget_mode": budget,
        "depth": depth,
        "provider_pref": provider,
        "detail_level": detail,
        "auto_approve_default": auto_approve_default,
        "cross_session_memory_default": cross_session_memory_default,
        "preference_text": preference_text,
    }
    save_cli_setup(setup)
    if input("Run provider key setup now? (y/N): ").strip().lower() in {"y", "yes"}:
        interactive_provider_setup()
    return setup
