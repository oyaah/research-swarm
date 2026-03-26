from __future__ import annotations

import hashlib
from typing import Any


class InMemoryQdrant:
    def __init__(self) -> None:
        self._docs: list[dict[str, Any]] = []

    def upsert(self, doc: dict[str, Any]) -> str:
        doc_id = hashlib.md5((doc.get("source_url", "") + doc.get("summary", "")).encode()).hexdigest()
        stored = {"id": doc_id, **doc}
        self._docs.append(stored)
        return doc_id

    def search(self, session_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        subset = [d for d in self._docs if d.get("session_id") == session_id]
        if not subset:
            return []
        query_terms = set(query.lower().split())

        def score(d: dict[str, Any]) -> int:
            text = (d.get("summary", "") + " " + d.get("title", "")).lower()
            return sum(1 for t in query_terms if t in text)

        ranked = sorted(subset, key=score, reverse=True)
        return ranked[:limit]


qdrant_store = InMemoryQdrant()
