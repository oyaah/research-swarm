from __future__ import annotations

from typing import Any

from services.swarm_engine.tracing import new_trace_id


def tool_result(status: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": status,
        "payload": payload,
        "trace_id": new_trace_id(),
    }
