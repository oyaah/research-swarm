from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

TRACE_LOG = Path("artifacts/traces.jsonl")
TRACE_LOG.parent.mkdir(parents=True, exist_ok=True)

try:  # pragma: no cover
    from langsmith import Client
except Exception:  # pragma: no cover
    Client = None

_LANGSMITH_CLIENT = None


def new_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex[:16]}"


def langsmith_env(api_key: str) -> None:
    global _LANGSMITH_CLIENT
    if api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", "research-swarm")
        if Client is not None and _LANGSMITH_CLIENT is None:
            try:
                _LANGSMITH_CLIENT = Client(api_key=api_key)
            except Exception:
                _LANGSMITH_CLIENT = None


def emit_trace(session_id: str, node_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    event = {
        "trace_id": new_trace_id(),
        "session_id": session_id,
        "node_name": node_name,
        "timestamp": int(time.time()),
        "payload": payload,
    }
    with TRACE_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=True) + "\n")

    if _LANGSMITH_CLIENT is not None:
        try:
            _LANGSMITH_CLIENT.create_run(
                name=node_name,
                run_type="chain",
                inputs={"session_id": session_id},
                outputs=payload,
                id=event["trace_id"],
                project_name=os.environ.get("LANGCHAIN_PROJECT", "research-swarm"),
                extra={"metadata": {"session_id": session_id, "node_name": node_name}},
            )
        except Exception:
            pass

    return event
