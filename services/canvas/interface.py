from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from services.canvas.vector_engine import NeuralCartographer
from services.logging import get_logger

if TYPE_CHECKING:
    from services.swarm_engine.llm import LLMAdapter

logger = get_logger("swarm.canvas")


async def generate_knowledge_map(
    session_id: str,
    user_query: str,
    evidence_items: list[Dict[str, Any]],
    output_dir: str,
    llm: "LLMAdapter | None" = None,
) -> str | None:
    """
    Orchestrates the generation of the semantic knowledge map.
    Returns the path to the generated .canvas file.
    """
    try:
        cartographer = NeuralCartographer()

        cluster_labels: dict[int, str] | None = None
        cluster_ids = None
        if llm is not None and not llm.use_mock:
            cluster_labels, cluster_ids = await _build_cluster_labels(cartographer, evidence_items, llm)

        canvas_data = cartographer.layout_evidence(
            evidence_items, user_query, cluster_labels, precomputed_cluster_ids=cluster_ids
        )

        if not canvas_data:
            return None

        path = Path(output_dir) / "knowledge_map.canvas"
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(canvas_data, f, indent=2)

        return str(path)

    except Exception as e:
        logger.error(f"Failed to generate knowledge map: {e}")
        return None


async def _build_cluster_labels(
    cartographer: NeuralCartographer,
    evidence_items: list[Dict[str, Any]],
    llm: "LLMAdapter",
) -> tuple[dict[int, str] | None, Any]:
    """Run K-Means, name each cluster via LLM. Returns (labels, cluster_ids) so the
    caller can pass cluster_ids to layout_evidence and skip a second KMeans fit."""
    try:
        import numpy as np
        from sklearn.cluster import KMeans

        valid = [i for i in evidence_items if i.get("embedding") and len(i["embedding"]) > 0]
        if len(valid) < 3:
            return None, None

        matrix = np.array([item["embedding"] for item in valid])
        n_clusters = min(5, max(2, int(np.sqrt(len(valid)))))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        cluster_ids = kmeans.fit_predict(matrix)

        clusters: dict[int, list[str]] = {}
        for item, cid in zip(valid, cluster_ids):
            clusters.setdefault(int(cid), []).append(item.get("title", ""))

        labels = await llm.label_clusters(clusters)
        return labels, cluster_ids
    except Exception as e:
        logger.warning(f"Cluster labeling failed, using defaults: {e}")
        return None, None
