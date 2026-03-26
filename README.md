# Research Swarm

Autonomous multi-agent research system with LLM-driven orchestration. Not a pipeline — an arena where AI agents operate with true agency.Works with both open source and close sourced models.


## What It Does

Submit a question. The swarm runs a coordinated research process:

1. **Planner** — breaks the question into a research plan with subquestions and tool assignments
2. **Researcher** — runs parallel web searches, fetches URLs, normalizes evidence
3. **Verifier** — scores each piece of evidence and flags contradictions
4. **Analyst** — decides whether evidence is sufficient or whether to loop back for more research
5. **Writer** — synthesizes a final report with citations
6. **Neural Cartographer** — generates an Obsidian-compatible `.canvas` knowledge map from the evidence

The Orchestrator is LLM-driven — it routes between agents dynamically based on current state and budget. No hardcoded `if/else` routing.

## Architecture

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

### Portfolio Mode

Runs two independent research lanes in parallel (groq_fast + groq_deep) then picks the best result via Pareto frontier selection. Each lane is prefixed in the CLI output so interleaved logs are distinguishable.

### Neural Cartographer

After research completes, evidence items are embedded (Together.xyz BAAI/bge-base-en-v1.5) and projected to a 2D canvas via PCA. K-Means clusters similar findings. A single LLM call names each cluster. Cosine-similarity edges connect nearest semantic neighbors. Output: `artifacts/reports/{session_id}/knowledge_map.canvas`. If Obsidian is installed, the CLI auto-opens it.

## Setup

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your API keys
```

Minimum required key: `GROQ_API_KEY` or `ANTHROPIC_API_KEY`.

Key settings:

```
GROQ_API_KEY=...
ANTHROPIC_API_KEY=...
SEARCHAPI_API_KEY=...          # searchapi.io (recommended)
TOGETHER_API_KEY=...           # for embeddings (Neural Cartographer)

MODEL_GROQ=openai/gpt-oss-120b
MODEL_GROQ_BACKUP=openai/gpt-oss-20b
MODEL_GROQ_KIMI_K2=moonshotai/kimi-k2-instruct
MODEL_ANTHROPIC=claude-sonnet-4-6
MODEL_ANTHROPIC_OPUS=claude-opus-4-6

# Budget tiers
MAX_STEPS_LOW=8
MAX_STEPS_BALANCED=20
MAX_STEPS_HIGH=50
```

### 3. Start the API server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

No Docker required. Qdrant is in-memory. Supabase is optional (for session persistence across restarts).

### 4. Run the CLI

```bash
python -m frontend.cli
```

First run starts a setup wizard and saves defaults to `~/.research_swarm/cli_setup.json`.

## CLI Usage

### Interactive mode

```bash
python -m frontend.cli
```

Each prompt shows your current settings:

```
mode: single  ·  budget: balanced  ·  depth: standard  ·  provider: auto
What do you want to research?
>
```

### Single query (non-interactive)

```bash
python -m frontend.cli --query "your question" --budget-mode high --depth deep
```

### Portfolio mode (2 parallel lanes)

```bash
python -m frontend.cli --mode portfolio --budget-mode high
```

In interactive portfolio mode, you're prompted to pick lanes before each query:

```
Portfolio lane: [1] both  [2] groq_fast  [3] groq_deep
```

### Inline overrides

Append `::key=value` to any query without changing your default settings:

```
Compare top LLM inference providers ::budget=high ::depth=deep ::provider=anthropic
```

Supported overrides: `::budget=`, `::depth=`, `::provider=`, `::lane=`, `::mode=`

### All flags

```
--query              Research question
--budget-mode        low | balanced | high  (default: balanced)
--depth              quick | standard | deep  (default: standard)
--mode               single | portfolio  (default: single)
--provider-pref      auto | mixed | groq | anthropic | together | gemini | openai
--lane-preference    auto | both | fast | deep
--detail-level       compact | brief | detail  (default: compact)
--auto-approve       Skip all HITL prompts
--cross-session-memory  Enable u-mem cross-session context
--resume-session     Attach to an existing session ID
--setup              Re-run setup wizard
--reset-setup        Clear saved defaults
```

## Keyboard Controls

Terminal is in cbreak mode during streaming:

| Key | Action |
|-----|--------|
| `Ctrl+O` | Detail mode (full rationale, model tags) |
| `Ctrl+B` | Compact mode |
| `steer: <msg>` + Enter | Inject steering message mid-run |
| `Ctrl+C` | Exit |

Type `detail`, `compact`, or `brief` + Enter to switch modes.

For HITL prompts, type your answer + Enter. Anything matching `"no preference"`, `"you decide"`, etc. auto-skips future HITL gates.

## Slash Commands

| Command | Action |
|---------|--------|
| `/sessions` | List active + persisted sessions |
| `/resume <id>` | Attach to an existing session |
| `/setup` | Provider key setup wizard |
| `/mode single\|portfolio` | Switch mode |
| `/quit` | Exit |

## Budget and Model Routing

| Mode | Planner | Researcher | Analyst | Writer | Max Steps |
|------|---------|------------|---------|--------|-----------|
| `low` | small Groq | small Groq | small Groq | mid Groq | 8 |
| `balanced` | large Groq | mixed scouts | large Groq | kimi/mid | 20 |
| `high` | Claude Opus | diverse Groq scouts | Claude Opus | Claude Sonnet | 50 |

The planner can override routing via `agent_config` in its response. Depth (`quick`/`standard`/`deep`) further scales researcher count and verifier passes.

**Provider fallback chain:** groq → anthropic → openai → together → gemini

**Search fallback chain:** searchapi.io → tavily → playwright/DuckDuckGo → Bing → Wikipedia

## HITL Gates

Three optional checkpoints, all skippable:

1. **Plan approval** — planner decides whether to ask based on query complexity
2. **Analysis checkpoint** — analyst can request clarification mid-research
3. **Draft review** — writer's draft before finalization

## Optional: Playwright Search

For JS-heavy pages and free-tier search:

```bash
./scripts/install_playwright.sh
```

Once installed, playwright is auto-prioritized in the search chain.

## Optional: Cross-Session Memory (u-mem)

Install [u-mem](https://github.com/oyaah/u-mem) and set `UMEM_SRC_PATH` if needed. Pass `--cross-session-memory` to recall context from previous sessions.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/research` | Start a session |
| `GET` | `/v1/stream/{id}` | SSE event stream |
| `GET` | `/v1/research/{id}` | Session status + final state |
| `POST` | `/v1/research/{id}/resume` | Submit HITL answer |
| `POST` | `/v1/research/{id}/steer` | Inject steering message |
| `GET` | `/v1/sessions` | List sessions |
| `GET` | `/v1/tools` | Tool schema |

## Tests

```bash
python -m pytest tests/ -q
```

## File Layout

```
api/                      FastAPI app, routes, models
services/swarm_engine/    LangGraph runtime, LLM adapter, state, config, embeddings
services/canvas/          Neural Cartographer (PCA → canvas, K-Means, cosine edges)
services/memory/          Cross-session memory adapter (u-mem)
tools/                    MCP-compatible tool implementations (search, fetch, qdrant)
frontend/                 Terminal CLI (cli.py, cli_events.py, cli_commands.py, cli_http.py)
agents/                   Per-agent persona definitions (*.AGENT.md)
scripts/                  install_playwright.sh, demo.sh, check_providers.py
infra/                    Supabase schema + docker-compose (optional)
tests/                    pytest test suite
```
