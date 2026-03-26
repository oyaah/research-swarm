# Research Swarm

Autonomous multi-agent research system with LLM-driven orchestration. Not a pipeline — an arena where AI agents operate with true agency.

## Architecture

The Orchestrator is the brain. Every node returns to the Orchestrator, which uses LLM reasoning to decide what happens next based on current state + budget remaining. No hardcoded if/else routing for agent decisions.

```
                    ┌──────────────┐
                    │ Orchestrator │ ← LLM-driven routing
                    └──────┬───────┘
       ┌─────────┬────────┼────────┬──────────┬──────────┐
       ▼         ▼        ▼        ▼          ▼          ▼
   Planner  Researcher  Verifier  Analyst   Writer   HITL nodes
       │         │        │        │          │          │
       └─────────┴────────┴────────┴──────────┴──────────┘
                    → back to Orchestrator
```

### File Layout

```
frontend/cli.py              — CLI entry point, SSE stream, cbreak keyboard input, Obsidian auto-open
frontend/cli_events.py       — Event rendering (compact / brief / detail modes), lane-prefixed portfolio output
frontend/cli_http.py         — HTTP client (start, stream, steer, resume)
frontend/cli_commands.py     — Slash commands (/sessions, /setup, /quit)
frontend/cli_setup.py        — First-time setup wizard (saves defaults to ~/.research-swarm/cli_setup.json)

services/swarm_engine/runtime.py      — LangGraph graph: orchestrator hub-and-spoke
services/swarm_engine/llm.py          — LLM adapter (plan, orchestrate, summarize, verify, write, analyst_decide, label_clusters)
services/swarm_engine/swarm_config.py — Budget-aware model routing + lane spawning
services/swarm_engine/state.py        — SwarmState TypedDict (all graph state fields)
services/swarm_engine/settings.py     — Pydantic-settings v2 config (.env)
services/swarm_engine/embeddings.py   — Together.xyz embedding adapter
services/swarm_engine/portfolio_engine.py — Multi-lane parallel execution + Pareto selection
services/swarm_engine/agent_personas.py   — Per-agent system prompt loading
services/swarm_engine/artifact_exports.py — Report export formats
services/swarm_engine/tracing.py      — Trace + LangSmith integration

services/canvas/interface.py      — Neural Cartographer orchestrator (async, accepts LLM for cluster labeling)
services/canvas/vector_engine.py  — PCA projection, K-Means clustering, cluster group nodes, cosine-similarity edges
services/memory/umem_adapter.py   — Cross-session memory via u-mem MCP

api/routes.py    — FastAPI: /v1/research, /v1/stream/{id}, /v1/research/{id}/steer, /v1/research/{id}/resume
api/models.py    — Request/response Pydantic models
tools/mcp_tools.py — web_search, open_url, playwright_fetch, qdrant, wikipedia, ask_user_question
agents/*.AGENT.md  — Per-agent persona definitions (planner, researcher, verifier, analyst, writer)
```

## Orchestrator (runtime.py)

Central `orchestrator_node` runs on every iteration:
1. Builds a state snapshot (evidence count, scores, budget, researcher_exhausted, last worker, etc.)
2. Calls `llm.orchestrate()` which returns `{next_action, rationale, focus_query}`
3. Applies safety rails (plan must exist, budget enforcement, researcher_exhausted handling)
4. Applies finalization rails (postprocess → save → metrics → end — strictly forward once final_markdown exists)
5. Routes to the chosen worker node; all workers return to orchestrator

**Safety rails (minimal by design):**
- No plan → force planner
- Budget exhausted → force writer or end
- `researcher_exhausted=True` + has evidence → force writer; no evidence → trigger one replan then write
- `postprocess` counts toward `post_final_iterations` cap (max 2 post-write loops)
- Dependency ordering: postprocess → save → metrics → end

**Researcher exhaustion detection (`runtime.py`):**
- `idx >= len(plan)` with no `focus_query` → `researcher_exhausted=True` immediately
- All query candidates are meta-text (via `_is_meta_query_text`) or already tried → `researcher_exhausted=True`
- Meta query filter uses `_META_QUERY_PATTERNS` (catches persona leaks, self-reflection text, etc.)

## Planner

`llm.plan()` returns both a research plan AND an `agent_config` object:
- Plan steps: `{id, description, subquestions, tools, expected_cost_tokens, priority, status}`
- Agent config: `{researchers, verifier_passes, min_verified_evidence, analyst_min_score, ask_hitl, model_routes, researcher_routes, search_results_per_query, url_fetch_limit, docs_per_subtask, writer_sections}`

The planner dynamically decides model routing, resource allocation, and HITL strategy per query.

**Query building (`_build_subtask_query`):** prefers `subquestions[0]` over concatenated `description`. Falls back through candidates in order, skipping anything matching `_META_QUERY_PATTERNS`. Always falls back to `user_query` as last resort.

## Event System

`runtime.emit(session_id, event)` → SSE → `cli_events.show_event()`

| Event type | Emitted by | Purpose |
|---|---|---|
| `plan_update` | planner | show step count + plan items |
| `reasoning_summary` | all agents | per-node display line |
| `thought_stream` | orchestrator | routing decision display |
| `orchestration_tree` | orchestrator | ASCII tree — **detail mode only** |
| `trace` | all agents | diagnostic/debug data |
| `evidence_item` | researcher | `◈ found <title>` |
| `hitl_request` | hitl nodes | pause for human input |
| `draft_markdown` / `final_markdown` | writer | suppressed during stream; final shown at end |
| `done` | run_session | elapsed + source count |
| `portfolio_*` | portfolio engine | lane status lines |
| `agent_message` | inter-agent | from→to messages (detail mode only) |

## Display Rules (cli_events.py)

**Compact mode** (default) — one line per agent pass:

| Agent | Icon | Display |
|---|---|---|
| Orchestrator | ⊛ | routing decision + rationale |
| Planner | ◆ | step count + model routing assignments |
| Researcher | ◈ | `"<search query>"` |
| Verifier | ◎ | `score 0.70` |
| Analyst | ◉ | `researching more` / `ready to write` |
| Writer | ✦ | `7 sources → writing` |

**Portfolio mode** — all lines prefixed with `[lane_id]` (dim) so interleaved output from parallel lanes is distinguishable.

**Detail mode** (`Ctrl+O`) — adds model tags, rationale, inter-agent messages, orchestration tree.

**Suppression rules:**
- Verifier `reasoning_summary` → ALWAYS suppressed (trace has the score)
- Analyst `reasoning_summary` → suppressed in compact (trace shows decision)
- Researcher subtask/batch traces → suppressed in compact/brief
- PostProcess, MetricsEmitter, SaveArtifact → suppressed in compact/brief
- `orchestration_tree` → suppressed in compact/brief (only shown in detail)
- `draft_markdown` / `final_markdown` → suppressed during stream

## Keyboard Controls

Terminal is in cbreak mode during streaming:

| Key | Action |
|---|---|
| `Ctrl+O` | Detail mode |
| `Ctrl+B` | Compact mode |
| `steer: <msg>` + Enter | Inject steering message |
| `Ctrl+C` | Exit |

HITL answers: type response + Enter.

**Inline overrides** (per query): `my question ::budget=high ::depth=deep ::mode=portfolio`

## Budget & Model Routing

Three tiers control model selection and step limits:

| Mode | Planner | Researcher | Analyst | Writer | Max Steps |
|---|---|---|---|---|---|
| `low` | small Groq | small Groq | small Groq | mid Groq | 8 |
| `balanced` | large Groq | mixed scouts | large Groq | kimi/mid | 20 |
| `high` | Claude Opus | diverse Groq scouts | Claude Opus | Claude Sonnet | 50 |

`swarm_config.py` builds initial routes; planner can override via `agent_config`.

## Provider Fallback

- All routes use `provider_locked: false` — graceful fallback when a provider is down
- `_chat_groq` has fast-fail: marks Groq dead on first 403, skips instantly after
- Anthropic has rate limiter: max 3 concurrent calls, 0.3s min interval, auto-retry on 429
- Fallback chain: groq → anthropic → openai → together → gemini

## Speed Architecture

All I/O-bound operations parallelized via `asyncio.gather`:
- Researcher: all `web_search()` queries run concurrently
- Researcher: all URL fetches per search batch run concurrently (cap at `url_fetch_limit`, default 20)
- Verifier: all evidence items scored concurrently
- Timeouts: 15s per URL fetch, 60s per evidence verification

Sync tools (`web_search`, `open_url`, `playwright_fetch`) wrapped in `run_in_executor`.

## Portfolio Mode

Runs multiple independent research lanes in parallel with different providers/strategies:
1. `build_portfolio_lanes()` creates lane configs
2. Each lane runs `run_session()` with `isolated_graph=True` and `autonomous_mode=True`
3. Lane events forwarded to parent session via `portfolio_lane_event` (with `lane_id` prefix in CLI)
4. `pareto_frontier()` + `choose_candidate()` selects best result
5. Steering messages broadcast to all active children

**Lane selection** (interactive): when mode=portfolio, CLI prompts `[1] both [2] groq_fast [3] groq_deep` before each query.

## Neural Cartographer

Spatial knowledge mapping — generates Obsidian-compatible `.canvas` files:
- Evidence items embedded into vector space (Together.xyz BAAI/bge-base-en-v1.5)
- PCA projects to 2D canvas coordinates
- K-Means clusters similar findings (n_clusters = min(5, max(2, √n)))
- Cluster group nodes visually bound each topic on the canvas (Obsidian group type)
- Single-pass LLM call (`llm.label_clusters`) names each cluster from member titles
- Cosine-similarity edges connect nearest semantic neighbors (threshold > 0.85)
- Output: `artifacts/reports/{session_id}/knowledge_map.canvas`
- CLI auto-opens in Obsidian (via `obsidian://open?path=...`) when Obsidian is installed

## HITL (Human-in-the-Loop)

Three HITL gates, all optional and skippable:
1. **Plan approval** — planner decides whether to ask via `ask_hitl` in agent_config
2. **Analysis checkpoint** — analyst can ask clarifying questions
3. **Draft review** — writer's draft can be reviewed before finalization

Defer patterns (`"no preference"`, `"you decide"`, etc.) auto-skip future HITL for that stage and enable `autonomous_mode`.

## Key Settings (`.env`)

```
GROQ_API_KEY=...
ANTHROPIC_API_KEY=...
SEARCHAPI_API_KEY=...
MODEL_GROQ=openai/gpt-oss-120b
MODEL_GROQ_BACKUP=openai/gpt-oss-20b
MODEL_GROQ_KIMI_K2=moonshotai/kimi-k2-instruct
MODEL_ANTHROPIC=claude-sonnet-4-6
MODEL_ANTHROPIC_OPUS=claude-opus-4-6
ANALYST_GOOD_SCORE=0.50
VERIFIER_SCORE_HIGH=0.60
MAX_STEPS_LOW=8
MAX_STEPS_BALANCED=20
MAX_STEPS_HIGH=50
```

## Running

```bash
# Start API server (no Docker needed — Qdrant is in-memory, Supabase is optional)
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Run CLI
python -m frontend.cli

# Single query
python -m frontend.cli --query "your question" --budget-mode balanced

# Portfolio mode (2 parallel lanes)
python -m frontend.cli --mode portfolio --budget-mode high

# High-depth single query with specific provider
python -m frontend.cli --query "your question" --budget-mode high --depth deep --provider-pref anthropic
```

## Testing

```bash
python -m pytest tests/ -q
```
