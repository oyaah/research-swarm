from __future__ import annotations

from typing import Any

try:  # pragma: no cover
    from supabase import Client, create_client
except Exception:  # pragma: no cover
    Client = object  # type: ignore[assignment]

    def create_client(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError("supabase SDK unavailable")

from services.swarm_engine.settings import settings


class SupabaseStore:
    def __init__(self) -> None:
        self.client: Client | None = None
        if settings.supabase_url and settings.supabase_key:
            self.client = create_client(settings.supabase_url, settings.supabase_key)

    def enabled(self) -> bool:
        return self.client is not None

    def upsert_session(self, session_id: str, status: str) -> None:
        if not self.client:
            return
        try:
            self.client.table("sessions").upsert({"id": session_id, "status": status}).execute()
        except Exception:
            return

    def list_recent_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.client:
            return []
        try:
            resp = self.client.table("sessions").select("id,status,updated_at").order("updated_at", desc=True).limit(limit).execute()
            rows = getattr(resp, "data", None) or []
            out: list[dict[str, Any]] = []
            for r in rows:
                out.append(
                    {
                        "session_id": str(r.get("id", "")),
                        "status": str(r.get("status", "unknown")),
                        "updated_at": str(r.get("updated_at", "")),
                        "source": "supabase",
                    }
                )
            return [r for r in out if r.get("session_id")]
        except Exception:
            return []

    def session_exists(self, session_id: str) -> bool:
        if not self.client:
            return False
        try:
            resp = self.client.table("sessions").select("id").eq("id", session_id).limit(1).execute()
            rows = getattr(resp, "data", None) or []
            return bool(rows)
        except Exception:
            return False

    def insert_checkpoint(self, row: dict[str, Any]) -> None:
        if not self.client:
            return
        try:
            self.client.table("checkpoints").insert(row).execute()
        except Exception:
            return


supabase_store = SupabaseStore()
