from __future__ import annotations

from typing import Any, Literal, TypedDict


class PlanItem(TypedDict):
    id: str
    description: str
    subquestions: list[str]
    tools: list[str]
    expected_cost_tokens: int
    priority: int
    status: Literal["pending", "done"]


class EvidenceItem(TypedDict):
    source_url: str
    title: str
    timestamp: str
    summary: str
    claims: list[str]
    verification_score: float
    contradiction: bool
    trace_id: str
    embedding: list[float]


class SwarmState(TypedDict, total=False):
    session_id: str
    user_query: str
    plan: list[PlanItem]
    pending_hitl: bool
    hitl_question: str
    hitl_answer: str
    plan_approved: bool
    review_feedback: str
    needs_revision: bool
    cycle_count: int
    current_plan_index: int
    budget_mode: str
    depth: str
    provider_pref: str
    task_type: str
    complexity: str
    research_batch_size: int
    verifier_passes: int
    min_verified_evidence: int
    source_hints: list[str]
    memory_context: str
    model_routes: dict[str, dict[str, str]]
    researcher_routes: list[dict[str, str]]
    cross_session_memory: bool
    focus_query: str
    interrupt_stage: str
    hitl_last_stage: str
    hitl_last_question: str
    hitl_opt_out_stages: list[str]
    autonomous_mode: bool
    analyst_min_score: float
    writer_sections: list[str]
    evidence_items: list[EvidenceItem]
    verification: dict[str, Any]
    final_evidence_set: list[EvidenceItem]
    provisional_evidence_set: list[EvidenceItem]
    missing_requirements: list[str]
    draft_markdown: str
    final_markdown: str
    needs_more_research: bool
    metrics: dict[str, float]
    artifacts: dict[str, str]
    search_results_per_query: int
    url_fetch_limit: int
    docs_per_subtask: int
    search_failures: list[dict[str, Any]]
    tool_registry: list[dict[str, Any]]
    orchestration_history: list[dict[str, Any]]
    orchestrator_next: str
    last_worker: str
    budget_steps_remaining: int
    budget_steps_used: int
    budget_exhausted: bool
    postprocessed: bool
    save_done: bool
    force_replan: bool
    verifier_focus_indices: list[int]
    post_final_iterations: int
    queries_asked: list[str]
    researcher_exhausted: bool
    replan_attempts: int
    tool_logs: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    error: str
