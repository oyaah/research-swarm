from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache

import httpx


@lru_cache(maxsize=1)
def _local_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@dataclass
class EmbeddingAdapter:
    api_key: str
    model: str
    use_mock: bool = True

    async def embed(self, text: str) -> list[float]:
        if self.use_mock:
            return [0.0] * 16

        # Use Together API if key is available
        if self.api_key:
            try:
                payload = {"model": self.model, "input": text[:8000]}
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                async with httpx.AsyncClient(timeout=60) as client:
                    res = await client.post("https://api.together.xyz/v1/embeddings", headers=headers, json=payload)
                    res.raise_for_status()
                    data = res.json()
                    emb = data.get("data", [{}])[0].get("embedding", [])
                    if isinstance(emb, list) and emb:
                        return [float(x) for x in emb]
            except Exception:
                pass  # fall through to local model

        # Local fallback: sentence-transformers (no API key required)
        try:
            model = _local_model()
            vec = await asyncio.get_event_loop().run_in_executor(
                None, lambda: model.encode(text[:8000], normalize_embeddings=True).tolist()
            )
            return vec
        except Exception as e:
            raise RuntimeError(f"Embedding failed (no Together API key and local model error): {e}") from e
