from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_name: str = "research-swarm"
    app_env: str = "dev"

    deepseek_api_key: str = Field(default="stub", alias="DEEPSEEK_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    searchapi_api_key: str = Field(default="stub", alias="SEARCHAPI_API_KEY")
    gemini_api_key: str = Field(default="stub", alias="GEMINI_API_KEY")
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    qdrant_url: str = Field(default="http://qdrant:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    supabase_url: str = Field(default="", alias="SUPABASE_URL")
    supabase_key: str = Field(default="", alias="SUPABASE_KEY")

    model_primary: str = Field(default="deepseek-ai/DeepSeek-R1", alias="MODEL_PRIMARY")
    model_fallback: str = Field(default="gemini-2.0-flash", alias="MODEL_FALLBACK")
    model_groq: str = Field(default="openai/gpt-oss-120b", alias="MODEL_GROQ")
    model_groq_backup: str = Field(default="openai/gpt-oss-20b", alias="MODEL_GROQ_BACKUP")
    model_groq_qwen32b: str = Field(default="", alias="MODEL_GROQ_QWEN32B")
    model_groq_llama_scout: str = Field(default="", alias="MODEL_GROQ_LLAMA_SCOUT")
    model_groq_kimi_k2: str = Field(default="", alias="MODEL_GROQ_KIMI_K2")
    model_openai: str = Field(default="gpt-4.1-mini", alias="MODEL_OPENAI")
    model_anthropic: str = Field(default="claude-sonnet-4-6", alias="MODEL_ANTHROPIC")
    model_anthropic_opus: str = Field(default="claude-opus-4-6", alias="MODEL_ANTHROPIC_OPUS")
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    source_hints_default: str = ""
    planner_search_results: int = Field(default=3, alias="PLANNER_SEARCH_RESULTS")
    planner_context_snippets: int = Field(default=3, alias="PLANNER_CONTEXT_SNIPPETS")
    search_provider_order: str = Field(default="tavily,playwright,searchapiio,duckduckgo", alias="SEARCH_PROVIDER_ORDER")
    search_query_suffix: str = Field(default="", alias="SEARCH_QUERY_SUFFIX")
    search_allow_final_duckduckgo: bool = Field(default=True, alias="SEARCH_ALLOW_FINAL_DUCKDUCKGO")
    search_playwright_timeout_ms: int = Field(default=30000, alias="SEARCH_PLAYWRIGHT_TIMEOUT_MS")
    search_allow_wikipedia_fallback: bool = Field(default=True, alias="SEARCH_ALLOW_WIKIPEDIA_FALLBACK")
    hitl_defer_patterns: str = Field(
        default=(
            r"\bno\s+opinion\b,"
            r"\bno\s+preference\b,"
            r"\bdo\s+whatever\b,"
            r"\bdo\s+what(?:ever)?\s+you\s+think\b,"
            r"\byou\s+decide\b,"
            r"\byour\s+call\b,"
            r"\bi\s+trust\s+you\b,"
            r"\bup\s+to\s+you\b,"
            r"\bno\s+strong\s+view\b"
        ),
        alias="HITL_DEFER_PATTERNS",
    )
    provider_order_auto_high: str = Field(default="anthropic,openai,groq,together,gemini", alias="PROVIDER_ORDER_AUTO_HIGH")
    provider_order_auto_balanced: str = Field(default="groq,anthropic,openai,together,gemini", alias="PROVIDER_ORDER_AUTO_BALANCED")
    provider_order_auto_low: str = Field(default="groq,gemini,together,openai,anthropic", alias="PROVIDER_ORDER_AUTO_LOW")
    provider_order_mixed_high: str = Field(default="anthropic,openai,groq,together,gemini", alias="PROVIDER_ORDER_MIXED_HIGH")
    provider_order_mixed_balanced: str = Field(default="groq,anthropic,openai,together,gemini", alias="PROVIDER_ORDER_MIXED_BALANCED")
    lane_hints_groq_fast: str = Field(default="latest,site:news.google.com", alias="LANE_HINTS_GROQ_FAST")
    lane_hints_groq_deep: str = Field(default="site:reuters.com,site:arxiv.org,site:sec.gov", alias="LANE_HINTS_GROQ_DEEP")
    lane_hints_together: str = Field(default="site:wikipedia.org,site:britannica.com", alias="LANE_HINTS_TOGETHER")
    lane_hints_gemini: str = Field(default="latest,site:google.com", alias="LANE_HINTS_GEMINI")
    lane_hints_openai: str = Field(default="site:openai.com,site:github.com", alias="LANE_HINTS_OPENAI")
    lane_hints_anthropic: str = Field(default="site:anthropic.com,site:github.com", alias="LANE_HINTS_ANTHROPIC")
    umem_src_paths: str = Field(default="", alias="UMEM_SRC_PATHS")
    umem_bootstrap_limit: int = Field(default=8, alias="UMEM_BOOTSTRAP_LIMIT")
    umem_bootstrap_min_importance: int = Field(default=2, alias="UMEM_BOOTSTRAP_MIN_IMPORTANCE")
    umem_bootstrap_budget_tokens: int = Field(default=1000, alias="UMEM_BOOTSTRAP_BUDGET_TOKENS")
    umem_search_min_importance: int = Field(default=2, alias="UMEM_SEARCH_MIN_IMPORTANCE")
    umem_save_max_chars: int = Field(default=3000, alias="UMEM_SAVE_MAX_CHARS")
    umem_default_tag: str = Field(default="research-swarm", alias="UMEM_DEFAULT_TAG")
    use_mock_llm: bool | None = Field(default=None, alias="USE_MOCK_LLM")
    hitl_timeout_seconds: int = 43200
    max_cycles: int = 3
    subtask_time_budget_seconds: int = 60
    max_plan_items: int = 4

    # Score thresholds — configurable so planner/env can tune without code changes
    # Relaxed defaults: prioritize content discovery over strict verification
    analyst_good_score: float = Field(default=0.50, alias="ANALYST_GOOD_SCORE")
    analyst_good_score_factoid: float = Field(default=0.35, alias="ANALYST_GOOD_SCORE_FACTOID")
    verifier_score_high: float = Field(default=0.60, alias="VERIFIER_SCORE_HIGH")
    verifier_score_low: float = Field(default=0.30, alias="VERIFIER_SCORE_LOW")

    # HITL approval keywords (comma-separated)
    hitl_approval_keywords: str = Field(
        default="approve,ok,yes,continue,looks good,proceed,finalize",
        alias="HITL_APPROVAL_KEYWORDS",
    )

    # Orchestrator step budgets per tier
    max_steps_low: int = Field(default=8, alias="MAX_STEPS_LOW")
    max_steps_balanced: int = Field(default=20, alias="MAX_STEPS_BALANCED")
    max_steps_high: int = Field(default=50, alias="MAX_STEPS_HIGH")

    # User-facing defaults
    default_thinking_depth: str = Field(default="standard", alias="DEFAULT_THINKING_DEPTH")
    default_report_length: str = Field(default="standard", alias="DEFAULT_REPORT_LENGTH")
    show_thinking: bool = Field(default=True, alias="SHOW_THINKING")

    checkpoint_db_path: str = "data/checkpoints.sqlite"
    report_dir: str = "artifacts/reports"
    snapshot_dir: str = "artifacts/snapshots"


settings = Settings()


def llm_mock_enabled() -> bool:
    if settings.use_mock_llm is not None:
        return settings.use_mock_llm
    has_live_provider = (
        settings.deepseek_api_key not in {"", "stub"}
        or settings.gemini_api_key not in {"", "stub"}
        or settings.groq_api_key not in {"", "stub"}
        or settings.openai_api_key not in {"", "stub"}
        or settings.anthropic_api_key not in {"", "stub"}
    )
    return not has_live_provider
