from __future__ import annotations

from typing import Any

import requests


def parse_sse_lines(resp: requests.Response):
    event_name = "message"
    data_lines: list[str] = []
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())
            continue
        if line == "":
            if data_lines:
                payload = "\n".join(data_lines)
                yield event_name, payload
            event_name = "message"
            data_lines = []


def start_session(api_base: str, body: dict[str, Any]) -> str:
    res = requests.post(f"{api_base}/v1/research", json=body, timeout=30)
    res.raise_for_status()
    return res.json()["session_id"]


def resume_session_http(api_base: str, session_id: str, answer: str) -> None:
    res = requests.post(f"{api_base}/v1/research/{session_id}/resume", json={"answer": answer}, timeout=30)
    res.raise_for_status()


def steer_session_http(api_base: str, session_id: str, message: str) -> None:
    res = requests.post(f"{api_base}/v1/research/{session_id}/steer", json={"message": message}, timeout=30)
    res.raise_for_status()


def fetch_status(api_base: str, session_id: str) -> dict[str, Any]:
    res = requests.get(f"{api_base}/v1/research/{session_id}", timeout=30)
    res.raise_for_status()
    return res.json()


def list_sessions_http(api_base: str) -> list[dict[str, str]]:
    res = requests.get(f"{api_base}/v1/sessions", timeout=30)
    res.raise_for_status()
    payload = res.json()
    return payload.get("sessions", [])
