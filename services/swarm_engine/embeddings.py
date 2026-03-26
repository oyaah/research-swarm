from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class EmbeddingAdapter:
    api_key: str
    model: str
    use_mock: bool = True

    async def embed(self, text: str) -> list[float]:
        if self.use_mock:
            return [0.0] * 16
        payload = {"model": self.model, "input": text[:8000]}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=60) as client:
            res = await client.post("https://api.together.xyz/v1/embeddings", headers=headers, json=payload)
            res.raise_for_status()
            data = res.json()
            emb = data.get("data", [{}])[0].get("embedding", [])
            if not isinstance(emb, list) or not emb:
                raise RuntimeError("Together embedding response missing vector")
            return [float(x) for x in emb]
