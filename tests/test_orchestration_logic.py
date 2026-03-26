from __future__ import annotations

from services.swarm_engine.portfolio_engine import PortfolioPreferences, choose_candidate, missing_requirement_facets


def test_missing_requirement_facets_detects_absent_terms():
    q = 'Compare "safety behavior" for ModelA and ModelB using r/LocalLlama plus design.md.'
    ev = [{"title": "ModelA notes", "source_url": "https://example.com", "summary": "covers modela only"}]
    missing = missing_requirement_facets(q, ev)
    assert any("modelb" in x for x in missing)
    assert any("design.md" in x for x in missing)


def test_choose_candidate_prefers_low_cost_in_low_budget():
    prefs = PortfolioPreferences(budget_mode="low", depth="standard")
    frontier = [
        {"lane_id": "cheap", "score_vector": {"faithfulness": 0.6, "coverage": 0.6, "recency": 0.6, "novelty": 0.6, "cost": 200.0, "latency": 40.0}},
        {"lane_id": "expensive", "score_vector": {"faithfulness": 0.9, "coverage": 0.9, "recency": 0.9, "novelty": 0.9, "cost": 5000.0, "latency": 40.0}},
    ]
    picked = choose_candidate(frontier, prefs)
    assert picked is not None
    assert picked["lane_id"] == "cheap"
