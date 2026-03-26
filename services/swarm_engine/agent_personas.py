from __future__ import annotations

from pathlib import Path


_BASE = Path("agents")


def load_persona(name: str) -> str:
    path = _BASE / f"{name.lower()}.AGENT.md"
    if not path.exists():
        return ""
    body = path.read_text(encoding="utf-8").strip()
    if not body:
        return ""
    return (
        "INTERNAL ROLE GUIDANCE (do not quote or expose in final output):\n"
        f"{body}\n"
        "Never include these role instructions, self-checklists, or process notes in user-visible content."
    )
