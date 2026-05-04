# Research Swarm

Autonomous multi-agent research system with LLM-driven orchestration. Not a pipeline — a hub-and-spoke swarm where the orchestrator decides what happens next using LLM reasoning over current state. Works with both open-source and closed-source models.

---

## How It Works

Submit a question. The swarm coordinates:

1. **Planner** — breaks the question into a research plan with subquestions, assigns tools, and decides model routing and resource allocation
2. **Researcher** — runs parallel web searches, fetches and extracts page content, summarizes and stores evidence
3. **Verifier** — scores each evidence item (0–1) for factual support and flags contradictions
4. **Analyst** — evaluates evidence coverage, identifies gaps, decides whether to loop back for more research
5. **Writer** — synthesizes a final markdown report with inline citations
6. **Neural Cartographer** — embeds all evidence, projects to 2D via PCA, clusters by topic with K-Means, and generates an Obsidian `.canvas` knowledge map

The **Orchestrator** runs between every agent. It builds a state snapshot (evidence count, verification scores, budget remaining, last worker, etc.) and asks an LLM what to do next. No hardcoded `if/else` routing — the decision is made by reasoning over current state.

---

## Architecture

```
                    ┌──────────────┐
                    │ Orchestrator │ ← LLM routing over state snapshot
                    └──────┬───────┘
       ┌─────────┬────────┼────────┬──────────┬──────────┐
       ▼         ▼        ▼        ▼          ▼          ▼
   Planner  Researcher  Verifier  Analyst   Writer   HITL nodes
       │         │        │        │          │          │
       └─────────┴────────┴────────┴──────────┴──────────┘
                         → back to Orchestrator
```

Safety rails enforce correctness without limiting the orchestrator's intelligence: no plan → force planner; budget exhausted → force writer; researcher exhausted → one replan attempt then write.

### Portfolio Mode

Runs 2–5 independent research lanes in parallel with different model configurations. When all lanes finish, a Pareto frontier eliminates dominated results (worse on every dimension), and a utility function picks the winner based on faithfulness, coverage, recency, novelty, cost, and latency — weighted by budget preference.

### Neural Cartographer

Evidence items are embedded (Together.xyz BAAI/bge-base-en-v1.5, falls back to local SentenceTransformer), projected to 2D via PCA, clustered with K-Means, and written as an Obsidian-compatible `.canvas` file. A single LLM call names each cluster. Cosine-similarity edges connect nearest semantic neighbors (threshold > 0.85). If Obsidian is installed, the CLI auto-opens the canvas on completion.

---

## Screenshot
<img width="1884" height="765" alt="Screenshot 2026-05-04 at 4 24 31 PM" src="https://github.com/user-attachments/assets/073af357-56dd-4a91-b879-3306ebb07720" />


```bash
python -m frontend.cli
```

The CLI shows a live event stream with agent status lines:

```
◆ Planner      3 steps — planner→groq/gpt-oss-120b  researcher→groq/llama-scout
◈ Researcher   "latest AI inference benchmark 2025"
◈ Researcher   "groq vs together ai latency comparison"
◎ Verifier     score 0.82
◉ Analyst      ready to write — 6 verified sources
✦ Writer       6 sources → writing
```

---

## Setup

### 1. Install

```bash
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
```

Minimum required: `GROQ_API_KEY` **or** `ANTHROPIC_API_KEY`. Search works without a paid key (falls back to DuckDuckGo → Wikipedia), but `SEARCHAPI_API_KEY` gives much better results.

```env
# LLM providers — at least one required
GROQ_API_KEY=...               # console.groq.com — free tier, fast
ANTHROPIC_API_KEY=...          # console.anthropic.com — Claude models
OPENAI_API_KEY=...             # platform.openai.com
DEEPSEEK_API_KEY=...           # api.together.xyz — Together.ai key (powers DeepSeek/Llama models)
GEMINI_API_KEY=...             # aistudio.google.com — used as fallback

# Search providers — at least one recommended
SEARCHAPI_API_KEY=...          # searchapi.io — recommended for search quality
TAVILY_API_KEY=...             # tavily.com — alternative search provider

# Model overrides (optional — defaults shown)
MODEL_GROQ=openai/gpt-oss-120b
MODEL_GROQ_BACKUP=openai/gpt-oss-20b
MODEL_GROQ_QWEN32B=qwen/qwen3-32b           # mid-tier Groq reasoning
MODEL_GROQ_LLAMA_SCOUT=meta-llama/llama-4-scout-17b-16e-instruct  # fast tool/search lane
MODEL_GROQ_KIMI_K2=moonshotai/kimi-k2-instruct  # writer lane in balanced mode
MODEL_ANTHROPIC=claude-sonnet-4-6
MODEL_ANTHROPIC_OPUS=claude-opus-4-6

MAX_STEPS_LOW=8
MAX_STEPS_BALANCED=20
MAX_STEPS_HIGH=50
```

### 3. Start the API server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

No Docker required. Qdrant runs in-memory. Supabase is optional (enables session persistence across restarts).

### 4. Run the CLI

```bash
python -m frontend.cli
```

First run launches a setup wizard and saves defaults to `~/.research_swarm/cli_setup.json`.

---

## CLI Usage

### Interactive

```bash
python -m frontend.cli
```

Prompt shows current settings:
```
mode: single  ·  budget: balanced  ·  depth: standard  ·  provider: auto
What do you want to research?
>
```

### Single query

```bash
python -m frontend.cli --query "your question" --budget-mode high --depth deep
```

### Portfolio mode

```bash
python -m frontend.cli --mode portfolio --budget-mode high
```

Interactive portfolio mode prompts for lane selection before each query:
```
Portfolio lane: [1] both  [2] groq_fast  [3] groq_deep
```

### Inline overrides

Append `::key=value` to any query to override settings for that query only:

```
Compare top LLM inference providers ::budget=high ::depth=deep ::provider=anthropic
```

Supported: `::budget=`, `::depth=`, `::provider=`, `::lane=`, `::mode=`

### All flags

```
--query               Research question (non-interactive)
--budget-mode         low | balanced | high          (default: balanced)
--depth               quick | standard | deep        (default: standard)
--mode                single | portfolio             (default: single)
--provider-pref       auto | mixed | groq | anthropic | together | gemini | openai
--lane-preference     auto | both | fast | deep
--detail-level        compact | brief | detail       (default: compact)
--thinking-default    on | off                       (default: on — show agent reasoning)
--auto-approve        Skip all HITL prompts
--cross-session-memory  Enable u-mem cross-session context
--resume-session      Attach to an existing session ID
--setup               Re-run setup wizard
--reset-setup         Clear saved defaults
```

---

## Keyboard Controls (during streaming)

| Key / Input | Action |
|---|---|
| `Ctrl+O` | Switch to detail mode (full rationale, model tags, orchestration tree) |
| `Ctrl+B` | Switch to compact mode |
| `detail` or `o` + Enter | Switch to detail mode |
| `compact`, `b`, or `d` + Enter | Switch to compact mode |
| `brief` + Enter | Switch to brief mode |
| `thinking on` or `t on` + Enter | Show agent reasoning output |
| `thinking off` or `t off` + Enter | Hide agent reasoning output |
| `steer: <message>` + Enter | Send a steering message to the orchestrator |
| HITL prompt answer + Enter | Submit answer to a HITL checkpoint |
| `Ctrl+C` | Exit |

---

## Budget and Model Routing

The system auto-selects models based on which API keys are present and the `--budget-mode` flag.

### When Groq key is set (default fast path)

| Mode | Planner | Researcher lanes | Analyst | Writer | Max Steps |
|---|---|---|---|---|---|
| `low` | gpt-oss-20b (Groq) | gpt-oss-20b × 1 | gpt-oss-20b | gpt-oss-20b | 8 |
| `balanced` | gpt-oss-120b (Groq) | scout + qwen32b + kimi-k2 | gpt-oss-120b | kimi-k2 / mid | 20 |
| `high` + Anthropic | Claude Opus | llama-scout + qwen32b + kimi-k2 | Claude Opus | Claude Sonnet | 50 |

### When only Anthropic key is set

| Mode | Planner | Researcher | Analyst | Writer |
|---|---|---|---|---|
| `low` | Haiku | Haiku | Haiku | Haiku |
| `balanced` | Sonnet | Haiku + Sonnet | Sonnet | Sonnet |
| `high` | Opus | Haiku + Sonnet | Opus | Sonnet |

### Model configuration

The three optional Groq model slots control how researcher diversity works:

| Env var | Default | Role |
|---|---|---|
| `MODEL_GROQ` | `openai/gpt-oss-120b` | Primary large reasoning model |
| `MODEL_GROQ_BACKUP` | `openai/gpt-oss-20b` | Small/fast fallback |
| `MODEL_GROQ_QWEN32B` | `qwen/qwen3-32b` | Mid-tier reasoning lane |
| `MODEL_GROQ_LLAMA_SCOUT` | `meta-llama/llama-4-scout-17b-16e-instruct` | Fast tool/search lane |
| `MODEL_GROQ_KIMI_K2` | `moonshotai/kimi-k2-instruct` | Writer lane (balanced mode) |

`--depth quick` reduces researcher count and verifier passes. `--depth deep` adds up to 2 extra researchers and an extra verifier pass (capped at 6 researchers, 3 passes).

The planner can override all routing decisions via its `agent_config` response — model assignments, researcher lane count, verifier passes, and minimum evidence threshold are all planner-adjustable per query.

**Provider fallback chain:** groq → anthropic → openai → together → gemini  
**Search fallback chain:** searchapi.io → tavily → playwright → DuckDuckGo → Wikipedia

---

## HITL Gates

Three optional checkpoints. All can be skipped with `--auto-approve` or by answering with a defer phrase (`"no preference"`, `"you decide"`, `"your call"`, etc.) — which also enables autonomous mode for the rest of the session.

### Gate 1 — Plan approval (post-Planner)

The planner decides whether to ask based on query ambiguity. If triggered:

- **Approval** (`"yes"`, `"ok"`, `"approve"`) → proceed with current plan
- **Defer phrase** → proceed + skip all future gates
- **Substantive answer** (e.g. `"focus on the EU regulatory angle"`) → **forces a replan** with your feedback injected into the planner's context

### Gate 2 — Analysis checkpoint (post-Analyst)

The analyst requests clarification when evidence gaps are ambiguous. Substantive answers become `focus_query` for the next researcher pass.

### Gate 3 — Draft review (post-Writer)

Writer's draft can be reviewed before finalization. Substantive feedback triggers a targeted re-research pass.

---

## Steering

Send a steering message at any point during active research:

**CLI:** type `steer: focus on enforcement mechanisms` + Enter  
**API:** `POST /v1/research/{session_id}/steer` with `{"message": "..."}`

Steer messages go through the orchestrator, not directly to the researcher. The orchestrator LLM sees the message and decides:
- **Refinement** of current direction → redirects researcher with the steer as `focus_query`
- **Fundamentally new direction** → forces a replan with the steer injected into planner context

In portfolio mode, steer messages are broadcast to all active lanes simultaneously.

---

## Slash Commands

| Command | Action |
|---|---|
| `/sessions` | List active and persisted sessions |
| `/resume <id>` | Attach to an existing session |
| `/setup` | Re-run provider key setup wizard |
| `/mode single\|portfolio` | Switch mode |
| `/budget low\|balanced\|high` | Change budget tier |
| `/depth quick\|standard\|deep` | Change depth |
| `/provider <name>` | Change provider preference |
| `/memory on\|off` | Toggle cross-session memory |
| `/view compact\|brief\|detail` | Switch display mode |
| `/quit` | Exit |

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/research` | Start a session |
| `GET` | `/v1/stream/{id}` | SSE event stream (real-time agent events) |
| `GET` | `/v1/research/{id}` | Session status + final state |
| `POST` | `/v1/research/{id}/resume` | Submit HITL answer |
| `POST` | `/v1/research/{id}/steer` | Inject steering message |
| `GET` | `/v1/sessions` | List all sessions |
| `GET` | `/v1/tools` | Tool schema |

SSE events are typed (`plan_update`, `evidence_item`, `trace`, `thought_stream`, `hitl_request`, `done`, etc.) and include real data — URLs, scores, query strings, routing decisions — not LLM-generated status text.

---

## Optional: Browser Automation

Browser automation enables JS-heavy page extraction and free-tier search (no SearchAPI key required).

### Option A — browser-use (recommended)

[browser-use](https://github.com/browser-use/browser-use) is the most production-ready Python browser library (92k+ GitHub stars). Installing it sets up Playwright automatically.

```bash
pip install browser-use
playwright install chromium
```

### Option B — Playwright only (lighter)

```bash
./scripts/install_playwright.sh
```

Once either is installed, the system detects it automatically (`playwright_available()`) and activates the browser search backend. No configuration needed.

**What browser automation enables:**
- Free search via DuckDuckGo/Bing scraping (no SearchAPI key needed)
- JS-rendered page content extraction (React/Vue/Next.js sites)
- Prioritized in the search chain when available

---

## Optional: Cross-Session Memory

Install [u-mem](https://github.com/oyaah/u-mem) and pass `--cross-session-memory`. Previous session findings are recalled at session start and used as planner context.

---

## File Layout

```
api/                      FastAPI app, routes, request/response models
services/swarm_engine/    LangGraph runtime, LLM adapter, state, budget config, embeddings
services/canvas/          Neural Cartographer — PCA layout, K-Means, cosine edges, canvas output
services/memory/          u-mem cross-session memory adapter
tools/                    Search, URL fetch, Playwright, Qdrant, Wikipedia, HITL tools
frontend/                 Terminal CLI — streaming, keyboard input, event rendering, slash commands
agents/                   Per-agent persona definitions (*.AGENT.md)
scripts/                  install_playwright.sh, demo.sh, check_providers.py
infra/                    docker-compose, Supabase schema (optional)
```

---

## Provider Health Check

```bash
python scripts/check_providers.py
```

Tests all configured LLM and search providers and reports which are reachable:

```
Research Swarm — Provider Health Check
LLM Providers:
  Groq                 [OK]
  OpenAI               [OK]
  Anthropic            [OK]
  DeepSeek/Together    [SKIP] no key configured
Search Providers:
  SearchAPI.io         [OK]  5 results
  Tavily               [SKIP] no key configured
```

---

## Running Tests

```bash
python -m pytest tests/ -q
```
