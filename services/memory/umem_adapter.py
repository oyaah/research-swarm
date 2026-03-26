from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

from services.swarm_engine.settings import settings


_UMEM_SERVER: Any | None = None
_UMEM_INIT_DONE = False


def _candidate_paths() -> list[Path]:
    env_path = os.getenv("UMEM_SRC_PATH", "").strip()
    env_paths = os.getenv("UMEM_SRC_PATHS", "").strip()
    repo_root = Path(__file__).resolve().parents[2]
    defaults = [
        repo_root / "u-mem" / "umem" / "src",
        Path.home() / "Desktop" / "u-mem" / "umem" / "src",
        Path.home() / "u-mem" / "umem" / "src",
    ]
    out: list[Path] = []
    if env_path:
        out.append(Path(env_path).expanduser())
    if env_paths:
        out.extend(Path(p.strip()).expanduser() for p in env_paths.split(",") if p.strip())
    if settings.umem_src_paths:
        out.extend(Path(p.strip()).expanduser() for p in settings.umem_src_paths.split(",") if p.strip())
    out.extend(defaults)
    dedup: list[Path] = []
    seen: set[str] = set()
    for p in out:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    return dedup


def _load_umem_server() -> Any | None:
    global _UMEM_SERVER, _UMEM_INIT_DONE
    if _UMEM_INIT_DONE:
        return _UMEM_SERVER
    _UMEM_INIT_DONE = True
    try:
        from umem import server as umem_server  # type: ignore

        _UMEM_SERVER = umem_server
        return _UMEM_SERVER
    except Exception:
        pass
    for candidate in _candidate_paths():
        if not candidate.exists():
            continue
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        try:
            from umem import server as umem_server  # type: ignore

            _UMEM_SERVER = umem_server
            return _UMEM_SERVER
        except Exception:
            continue
    _UMEM_SERVER = None
    return None


def mem_bootstrap_pack(query: str, project: str = "research-swarm") -> dict[str, Any]:
    srv = _load_umem_server()
    if srv is None:
        return {"status": "unavailable", "payload": {"error": "u-mem not found"}}
    try:
        out = srv.mem_bootstrap(
            project=project,
            focus_query=query,
            limit=max(1, int(settings.umem_bootstrap_limit)),
            min_importance=max(0, int(settings.umem_bootstrap_min_importance)),
            budget_tokens=max(128, int(settings.umem_bootstrap_budget_tokens)),
        )
        return {"status": "ok", "payload": out}
    except Exception as exc:
        return {"status": "error", "payload": {"error": str(exc)}}


def mem_search_pack(query: str, project: str = "research-swarm", limit: int = 5) -> dict[str, Any]:
    srv = _load_umem_server()
    if srv is None:
        return {"status": "unavailable", "payload": {"error": "u-mem not found"}}
    try:
        out = srv.mem_search(
            query=query,
            project=project,
            limit=max(1, int(limit)),
            min_importance=max(0, int(settings.umem_search_min_importance)),
            include_stale=False,
        )
        return {"status": "ok", "payload": out}
    except Exception as exc:
        return {"status": "error", "payload": {"error": str(exc)}}


def mem_save_note(content: str, tags: list[str] | None = None, project: str = "research-swarm", importance: int = 3) -> dict[str, Any]:
    srv = _load_umem_server()
    if srv is None:
        return {"status": "unavailable", "payload": {"error": "u-mem not found"}}
    try:
        out = srv.mem_save(
            content=content[: max(200, int(settings.umem_save_max_chars))],
            importance=importance,
            tags=tags or [settings.umem_default_tag],
            project=project,
        )
        return {"status": "ok", "payload": out}
    except Exception as exc:
        return {"status": "error", "payload": {"error": str(exc)}}
