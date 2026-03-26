from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

# Obsidian Canvas color codes: 1=Red 2=Orange 3=Yellow 4=Green 5=Cyan 6=Purple
_CLUSTER_COLORS = ["4", "3", "6", "5", "2"]   # green, yellow, purple, cyan, orange
_GROUP_PADDING = 100                            # px around node bounding box for groups


@dataclass
class CanvasNode:
    id: str
    text: str
    x: int
    y: int
    width: int = 400
    height: int = 200
    color: str = "1"
    type: str = "text"


@dataclass
class CanvasEdge:
    id: str
    from_node: str
    to_node: str
    color: Optional[str] = None
    to_side: str = "left"


@dataclass
class NeuralCartographer:
    """Converts text evidence into a spatial semantic Obsidian Canvas map."""

    scale_factor: int = 2000

    def layout_evidence(
        self,
        evidence_items: List[Dict[str, Any]],
        user_query: str,
        cluster_labels: Optional[Dict[int, str]] = None,
        precomputed_cluster_ids: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Takes evidence with embeddings, returns JSON Canvas dict."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not found. Neural Cartographer disabled.")
            return {}

        valid_items = [i for i in evidence_items if i.get("embedding") and len(i["embedding"]) > 0]
        if len(valid_items) < 3:
            logger.info("Not enough evidence for spatial layout (<3 items).")
            return {}

        matrix = np.array([item["embedding"] for item in valid_items])

        # PCA: high-dim embeddings → 2D canvas coordinates
        pca = PCA(n_components=2)
        coords = pca.fit_transform(matrix)
        coords = coords - coords.mean(axis=0)
        max_val = np.abs(coords).max()
        if max_val > 0:
            coords = (coords / max_val) * self.scale_factor

        # K-Means clustering (skip if caller already computed)
        if precomputed_cluster_ids is not None:
            cluster_ids = precomputed_cluster_ids
        else:
            n_clusters = min(5, max(2, int(np.sqrt(len(valid_items)))))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            cluster_ids = kmeans.fit_predict(matrix)

        # ── Build evidence nodes ──────────────────────────────────────────────
        evidence_nodes: List[CanvasNode] = []
        cluster_node_map: Dict[int, List[CanvasNode]] = {}

        for idx, item in enumerate(valid_items):
            x, y = coords[idx]
            # Push nodes away from the root area at the center
            if abs(x) < 350 and abs(y) < 200:
                x += 450 if x >= 0 else -450

            cid = int(cluster_ids[idx])
            color_code = _CLUSTER_COLORS[cid % len(_CLUSTER_COLORS)]
            cluster_label = (cluster_labels or {}).get(cid, f"Topic {cid + 1}")

            title = item.get("title") or "Source"
            short_title = title.split(".")[0][:80]

            summary = item.get("summary", "")
            first_sentence = summary.split(".")[0].strip()[:120]
            if first_sentence and not first_sentence.endswith("."):
                first_sentence += "."

            score = item.get("verification_score", 0.0)
            score_bar = "●" * round(score * 5) + "○" * (5 - round(score * 5))

            node = CanvasNode(
                id=f"node_{idx}",
                text=f"**{short_title}**\n\n{first_sentence}\n\n*{cluster_label}* · {score_bar}",
                x=int(x),
                y=int(y),
                width=360,
                height=160,
                color=color_code,
            )
            evidence_nodes.append(node)
            cluster_node_map.setdefault(cid, []).append(node)

        # ── Build cluster group nodes ─────────────────────────────────────────
        # Groups render behind their member nodes, visually bounding each topic.
        group_nodes: List[CanvasNode] = []
        for cid, members in cluster_node_map.items():
            if len(members) < 2:
                continue
            min_x = min(n.x for n in members) - _GROUP_PADDING
            min_y = min(n.y for n in members) - _GROUP_PADDING
            max_x = max(n.x + n.width for n in members) + _GROUP_PADDING
            max_y = max(n.y + n.height for n in members) + _GROUP_PADDING
            label = (cluster_labels or {}).get(cid, f"Topic {cid + 1}")
            group_nodes.append(CanvasNode(
                id=f"group_{cid}",
                text=label,
                x=min_x,
                y=min_y,
                width=max_x - min_x,
                height=max_y - min_y,
                color=_CLUSTER_COLORS[cid % len(_CLUSTER_COLORS)],
                type="group",
            ))

        # ── Root query node ───────────────────────────────────────────────────
        root = CanvasNode(
            id="root_query",
            text=f"## {user_query}",
            x=-250,
            y=-150,
            width=500,
            height=120,
            color="1",
        )

        # ── Edges: one per node to its nearest semantic neighbour ─────────────
        edges: List[CanvasEdge] = []
        seen_edges: set[tuple[int, int]] = set()
        sim_matrix = cosine_similarity(matrix)

        for i in range(len(valid_items)):
            sims = sim_matrix[i].copy()
            sims[i] = -1.0
            j = int(np.argmax(sims))
            if float(sims[j]) > 0.85:
                pair = (min(i, j), max(i, j))
                if pair not in seen_edges:
                    seen_edges.add(pair)
                    edges.append(CanvasEdge(
                        id=f"edge_{pair[0]}_{pair[1]}",
                        from_node=f"node_{i}",
                        to_node=f"node_{j}",
                    ))

        # ── Serialize ────────────────────────────────────────────────────────
        # Groups first so they render behind text nodes in Obsidian.
        serialized_nodes = []
        for n in group_nodes:
            serialized_nodes.append({
                "id": n.id, "type": "group", "label": n.text,
                "x": n.x, "y": n.y, "width": n.width, "height": n.height, "color": n.color,
            })
        serialized_nodes.append({
            "id": root.id, "type": root.type, "text": root.text,
            "x": root.x, "y": root.y, "width": root.width, "height": root.height, "color": root.color,
        })
        for n in evidence_nodes:
            serialized_nodes.append({
                "id": n.id, "type": n.type, "text": n.text,
                "x": n.x, "y": n.y, "width": n.width, "height": n.height, "color": n.color,
            })

        return {
            "nodes": serialized_nodes,
            "edges": [
                {"id": e.id, "fromNode": e.from_node, "toNode": e.to_node,
                 "color": e.color, "toSide": e.to_side}
                for e in edges
            ],
        }
