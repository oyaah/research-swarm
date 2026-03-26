from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
import asyncio

import httpx

from services.swarm_engine.agent_personas import load_persona


class _RateLimiter:
    """Simple token-bucket rate limiter for API calls."""
    def __init__(self, max_concurrent: int = 3, min_interval: float = 0.5):
        self._sem = asyncio.Semaphore(max_concurrent)
        self._min_interval = min_interval
        self._last_call = 0.0

    async def acquire(self):
        await self._sem.acquire()
        now = asyncio.get_event_loop().time()
        wait = self._min_interval - (now - self._last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call = asyncio.get_event_loop().time()

    def release(self):
        self._sem.release()


_anthropic_limiter = _RateLimiter(max_concurrent=3, min_interval=0.3)


@dataclass
class LLMAdapter:
    use_mock: bool = True
    deepseek_api_key: str = "stub"
    gemini_api_key: str = "stub"
    groq_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    model_primary: str = "deepseek-ai/DeepSeek-R1"
    model_fallback: str = "gemini-2.0-flash"
    model_groq: str = "openai/gpt-oss-120b"
    model_groq_backup: str = "openai/gpt-oss-20b"
    model_groq_qwen32b: str = ""
    model_groq_llama_scout: str = ""
    model_groq_kimi_k2: str = ""
    model_openai: str = "gpt-4.1-mini"
    model_anthropic: str = "claude-sonnet-4-6"
    model_anthropic_opus: str = "claude-opus-4-6"

    async def plan(
        self,
        query: str,
        route: dict[str, str] | None = None,
        context: str = "",
        available_models: str = "",
        available_tools: str = "",
    ) -> tuple[list[dict], dict]:
        if self.use_mock:
            return self._mock_plan(query), {}

        models_block = f"\n\n{available_models}" if available_models else ""
        tools_block = f"\n\nAvailable tools:\n{available_tools}" if available_tools else ""
        prompt = (
            load_persona("planner")
            + models_block
            + tools_block
            + "\n\n"
            "Return a JSON OBJECT (not array) with exactly two top-level keys:\n"
            "\n"
            "1. \"plan\" — array of research steps. Structure the plan as you see fit for this query:\n"
            "   - For simple/factoid queries: a lean 1-2 step retrieval-and-verify plan is fine.\n"
            "   - For analytical/complex queries: consider staged exploration before verification, "
            "multiple subtasks with different angles, and a synthesis step.\n"
            "   - There is no required phase structure — choose what fits the query complexity.\n"
            "   Each step fields: id, description, subquestions (array), tools (array), "
            "expected_cost_tokens (int), priority (int), status (string, always 'pending').\n"
            "\n"
            "2. \"agent_config\" — your behavioral decisions for this query:\n"
            "   - researchers (int 1-4): parallel researcher lanes to spawn\n"
            "   - verifier_passes (int 1-3): verification rounds per evidence item\n"
            "   - min_verified_evidence (int 1-5): minimum quality evidence items before synthesizing\n"
            "   - analyst_min_score (float 0-1): score threshold for 'good enough' evidence — "
            "lower for factoid queries (~0.45), higher for analytical (~0.65-0.8)\n"
            "   - ask_hitl (bool): whether to pause for human approval after planning\n"
            "   - hitl_question (string): the clarifying question to ask if ask_hitl is true\n"
            "   - writer_sections (array of strings): report section headers the writer should use — "
            "choose based on query type (e.g. [\"Summary\", \"Findings\", \"Analysis\", \"Conclusion\"] "
            "or [\"Answer\", \"Sources\"] for factoid); empty array means writer decides\n"
            "   - model_routes (object): one entry per role — planner, researcher, verifier, analyst, writer\n"
            "     Each value: {\"provider\": \"<provider>\", \"model\": \"<model_id>\"}\n"
            "   - researcher_routes (array): [{\"provider\": \"<p>\", \"model\": \"<m>\"}] — "
            "one per researcher lane, use DIFFERENT models for diversity\n"
            "   - search_results_per_query (int 5-15): how many search results to fetch per query — "
            "use 5-6 for factoid, 10-15 for deep analytical/comparison queries\n"
            "   - url_fetch_limit (int 10-30): max URLs to fetch and read per subtask — "
            "higher for broad surveys, lower for focused lookups\n"
            "   - docs_per_subtask (int 3-12): max evidence documents to extract per subtask — "
            "higher for comparison/survey queries, lower for factoid\n"
            "   - rationale (string): one sentence explaining your key choices\n"
            "\n"
            "Use fast/small models for retrieval lanes, large/reasoning models for analysis and writing.\n"
            "Do not include meta/self-referential text in plan fields. "
            "Never output persona commentary, internal monologue, or 'as an AI' style language in descriptions/subquestions.\n"
            f"\nQuery: {query}"
            + ("\nRecent context:\n" + context if context else "")
        )
        last_text = ""
        for attempt in range(3):
            text = await self._chat(prompt, temperature=0.1, route=route)
            last_text = text
            obj = self._parse_json_object(text)
            if obj and "plan" in obj:
                plan_items = obj.get("plan", [])
                if isinstance(plan_items, list) and plan_items:
                    agent_config = self._normalize_agent_config(obj.get("agent_config") or {})
                    return plan_items, agent_config
            loose_obj = self._parse_json_object_loose(text)
            if loose_obj and "plan" in loose_obj and isinstance(loose_obj.get("plan"), list) and loose_obj.get("plan"):
                agent_config = self._normalize_agent_config(loose_obj.get("agent_config") or {})
                return loose_obj.get("plan", []), agent_config
            # Might have returned an array directly (old-style planner response)
            parsed = self._parse_json_array(text)
            if parsed:
                return parsed, {}
            coerced = self._coerce_plan_from_text(text, query)
            if coerced:
                return coerced, {}
            prompt = "Return ONLY a valid JSON object with keys 'plan' and 'agent_config'. " + prompt
        raise RuntimeError(f"Live planner failed to return valid JSON plan. Raw: {last_text[:500]}")

    async def orchestrate(
        self,
        query: str,
        state_snapshot: dict[str, Any],
        allowed_actions: list[str],
        route: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if self.use_mock:
            if state_snapshot.get("pending_hitl"):
                return {"next_action": "end_waiting", "rationale": "awaiting human input"}
            if not state_snapshot.get("has_plan"):
                return {"next_action": "planner", "rationale": "need an execution plan"}
            if state_snapshot.get("has_final_markdown"):
                if not state_snapshot.get("postprocessed", False):
                    return {"next_action": "postprocess", "rationale": "final draft exists; postprocess first"}
                if not state_snapshot.get("save_done", False):
                    return {"next_action": "save", "rationale": "persist report artifacts"}
                if not state_snapshot.get("has_metrics", False):
                    return {"next_action": "metrics", "rationale": "compute final metrics"}
                return {"next_action": "end", "rationale": "session complete"}
            lw = state_snapshot.get("last_worker", "")
            ev = state_snapshot.get("evidence_count", 0)
            has_v = state_snapshot.get("has_verification", False)
            if lw == "" or lw == "planner" or lw == "hitl":
                return {"next_action": "researcher", "rationale": "execute plan subtasks"}
            if lw == "researcher" and not has_v:
                return {"next_action": "verifier", "rationale": "verify collected evidence"}
            if lw == "verifier" or lw == "analyst":
                if state_snapshot.get("needs_more_research") and not state_snapshot.get("budget_exhausted"):
                    return {"next_action": "researcher", "rationale": "analyst wants more evidence"}
                return {"next_action": "writer", "rationale": "evidence sufficient, synthesize"}
            if lw == "researcher" and has_v:
                return {"next_action": "analyst", "rationale": "re-analyze after additional research"}
            return {"next_action": "writer", "rationale": "synthesize report"}

        prompt = (
            "You are the central orchestrator of a multi-agent research swarm.\n"
            "Your job: decide what happens NEXT to best answer the user's query.\n\n"
            "Available actions and what they do:\n"
            "- researcher: Search the web, fetch URLs, extract evidence. Use when you need MORE data.\n"
            "- verifier: Score each evidence item for reliability and contradictions. Use AFTER research.\n"
            "- analyst: Analyze evidence quality, decide if coverage is sufficient, identify gaps.\n"
            "- writer: Synthesize a final report from verified evidence. Use when evidence is SUFFICIENT.\n"
            "- hitl_review: Ask user to review the draft. Use after writing if the query is ambiguous.\n\n"
            "Decision guidelines:\n"
            "- After planner: always route to researcher first\n"
            "- After researcher: verify evidence before writing (unless trivial factoid with 1-2 clear sources)\n"
            "- After verifier: route to analyst to assess gaps, or straight to writer if coverage is clearly good\n"
            "- After analyst with needs_more_research=true: route back to researcher with refined focus\n"
            "- After analyst with needs_more_research=false: route to writer\n"
            "- If avg_verification_score > 0.7 and evidence_count >= 5: likely ready to write\n"
            "- If budget is low: prioritize writing over perfecting evidence\n"
            "- Do NOT loop researcher→verifier endlessly if scores aren't improving\n\n"
            "Return JSON with keys: next_action (string), rationale (string), "
            "focus_query (string, optional refined search query), "
            "research_batch_size (int, optional), verify_indices (array<int>, optional).\n"
            f"Allowed next_action values: {allowed_actions}\n\n"
            f"User query: {query}\n"
            f"Current state: {json.dumps(state_snapshot)[:4000]}"
        )
        text = await self._chat(prompt, temperature=0.0, route=route)
        obj = self._parse_json_object(text)
        action = str(obj.get("next_action", "")).strip()
        if action not in allowed_actions:
            action = "researcher" if "researcher" in allowed_actions else allowed_actions[0]
        batch_size: int | None = None
        raw_batch = obj.get("research_batch_size")
        if isinstance(raw_batch, (int, float)):
            batch_size = int(raw_batch)
        verify_indices: list[int] = []
        raw_verify = obj.get("verify_indices")
        if isinstance(raw_verify, list):
            for x in raw_verify:
                if isinstance(x, (int, float)):
                    verify_indices.append(int(x))
        return {
            "next_action": action,
            "rationale": str(obj.get("rationale", "")).strip()[:500],
            "focus_query": str(obj.get("focus_query", "")).strip()[:300],
            "research_batch_size": batch_size,
            "verify_indices": verify_indices,
        }

    async def summarize(self, text: str, route: dict[str, str] | None = None) -> tuple[str, list[str]]:
        if self.use_mock:
            words = text.split()
            summary = " ".join(words[:80])
            claims = ["Claim: " + " ".join(words[:14]) if words else "Claim: no content"]
            return summary, claims

        prompt = (
            load_persona("researcher")
            + "\n\n"
            "Summarize in 3-5 sentences capturing the most important insights, data points, and conclusions. "
            "Extract up to 5 specific factual claims with numbers, dates, or concrete details. "
            "Return JSON object with keys summary and claims (array). Text:\n" + text[:10000]
        )
        for _ in range(3):
            text_out = await self._chat(prompt, temperature=0.2, route=route)
            obj = self._parse_json_object(text_out)
            if obj:
                summary = str(obj.get("summary", "")).strip()
                claims = obj.get("claims", [])
                claims = claims if isinstance(claims, list) else []
                cleaned_claims = [str(c).strip() for c in claims if str(c).strip()][:5]
                if summary:
                    return summary, (cleaned_claims or ["Insufficient structured claims extracted from source text."])
            coerced = self._coerce_summary_from_text(text_out)
            if coerced:
                return coerced
            prompt = "Return STRICT JSON object only: {\"summary\": string, \"claims\": [string, string, string]}.\n" + prompt
        raise RuntimeError("Live summarizer failed: unable to extract summary/claims")

    async def verify(
        self,
        query: str,
        title: str,
        summary: str,
        claims: list[str],
        source_excerpt: str = "",
        route: dict[str, str] | None = None,
    ) -> tuple[float, bool, str]:
        if self.use_mock:
            contradiction = "counterpoint" in title.lower() or "contradict" in summary.lower()
            return (0.82 if contradiction else 0.9), contradiction, "mock_verifier"

        has_source = bool(source_excerpt and source_excerpt.strip())
        source_block = (
            f"\n\nSOURCE EXCERPT (raw page text — ground truth):\n{source_excerpt[:2500]}"
            if has_source
            else "\n\nSOURCE EXCERPT: [unavailable — fetching failed or URL unreachable]"
        )
        prompt = (
            load_persona("verifier")
            + "\n\nReturn JSON only with REQUIRED keys: "
            "verification_score (0..1 number), contradiction (boolean), rationale (string), "
            "supported_claims (array of claim strings), missing_evidence (array of strings), citations (array of urls). "
            "\n\nSCORING RULES: "
            "verification_score > 0.6 if claims are supported by the SOURCE EXCERPT or are factually plausible. "
            "verification_score 0.35-0.6 if claims are plausible but not directly confirmed. "
            "verification_score < 0.35 ONLY if claims clearly contradict the excerpt or are obviously false. "
            "If SOURCE EXCERPT is unavailable, score based on claim plausibility (default 0.45, not penalized). "
            "Set contradiction=true ONLY if the excerpt contains statements that directly oppose the claims."
            f"\n\nQuestion: {query}\nTitle: {title}\nSummary: {summary}\nClaims: {claims}"
            + source_block
        )
        retries = 3
        last_error = "unknown"
        for _ in range(retries):
            text = await self._chat(prompt, temperature=0.0, route=route)
            try:
                obj = self._parse_json_object(text)
                normalized = self._normalize_verifier_object(obj, claims=claims, raw_text=text)
                if not normalized:
                    normalized = self._coerce_verification_from_text(text, claims=claims)
                if not normalized:
                    raise ValueError("verifier json missing required keys")
                score = float(normalized.get("verification_score", 0.35))
                contradiction = bool(normalized.get("contradiction", False))
                rationale = str(normalized.get("rationale", "")).strip() or "verifier_schema_normalized"
                return max(0.0, min(1.0, score)), contradiction, rationale
            except Exception as exc:
                last_error = str(exc)
                prompt = "Fix output to valid JSON schema only. " + prompt
        # Conservative fallback: do not fail the whole lane due to schema drift.
        return 0.35, False, f"verifier_schema_fallback: {last_error}"

    async def write(
        self,
        query: str,
        evidence: list[dict],
        route: dict[str, str] | None = None,
        sections: list[str] | None = None,
    ) -> tuple[str, str]:
        if self.use_mock:
            cites = [f"[{idx}]({item['source_url']})" for idx, item in enumerate(evidence, start=1)]
            body = "\n".join(f"- {item['summary']}" for item in evidence[:8])
            draft = (
                f"# Research Report\n\n## Question\n{query}\n\n"
                f"## Evidence\n{body or '- No evidence collected.'}\n\n"
                f"## Preliminary Citations\n" + "\n".join(cites)
            )
            final = draft + "\n\n## Bibliography\n" + "\n".join(cites)
            return draft, final

        evidence_bullets = "\n".join(
            f"- {i+1}. {e.get('summary','')} (source: {e.get('source_url','')})" for i, e in enumerate(evidence[:12])
        )
        if sections:
            section_list = ", ".join(sections)
            sections_instruction = f"Organize your report using these sections: {section_list}. "
        else:
            sections_instruction = (
                "Write a comprehensive, deeply analytical markdown report that directly answers the question. "
                "Go beyond surface-level summaries — reason about WHY things are the way they are, "
                "identify non-obvious patterns, connect disparate evidence, and offer original analysis. "
                "Structure the report however best fits the topic. Do NOT use a rigid template. "
                "Include a section on gaps, uncertainties, and what remains unknown. "
                "End with a Bibliography section listing all sources. "
            )
        prompt = (
            load_persona("writer")
            + "\n\n"
            + sections_instruction
            + "Output only the final report markdown. "
            + "Do not include planning notes, self-audit checklists, rewrite instructions, or meta commentary.\n"
            + "Use inline citations [n](url) only from provided evidence.\n"
            f"Question: {query}\nEvidence:\n{evidence_bullets}"
        )
        draft = await self._chat(prompt, temperature=0.3, route=route, max_completion_tokens=3600)
        if not draft.strip():
            raise RuntimeError("Live writer returned empty response")
        final = draft
        # Strip accidental meta-instruction leakage from model output.
        leak_markers = (
            "we need to produce",
            "before finalizing",
            "self-audit",
            "rewrite once for precision",
            "rewrite once for readability",
            "requested but not found",
        )
        low = final.lower()
        for marker in leak_markers:
            idx = low.find(marker)
            if idx != -1:
                final = final[:idx].rstrip()
                break
        if not final.strip():
            final = draft
        return draft, final

    async def decide_hitl(
        self,
        stage: str,
        query: str,
        context: str,
        route: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if self.use_mock:
            ask = stage == "plan" and any(k in query.lower() for k in ["evaluate", "design", "compare", "strategy"])
            question = "Anything to clarify before we continue?"
            return {"ask_user": ask, "question": question}

        prompt = (
            "Return JSON only with keys: ask_user (boolean), question (string), rationale (string). "
            "Decide if this is the right moment to ask user clarification, based on ambiguity/risk/context gaps.\n"
            f"Stage: {stage}\nUser query: {query}\nContext:\n{context[:2500]}"
        )
        text = await self._chat(prompt, temperature=0.0, route=route)
        obj = self._parse_json_object(text)
        ask_user = bool(obj.get("ask_user", False))
        question = str(obj.get("question", "")).strip()
        if ask_user and not question:
            question = "What should we prioritize next?"
        return {"ask_user": ask_user, "question": question}

    async def analyst_decide(
        self,
        query: str,
        evidence_items: list[dict[str, Any]],
        cycle_count: int,
        max_cycles: int,
        min_verified_evidence: int,
        route: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if self.use_mock:
            strong = [i for i in evidence_items if float(i.get("verification_score", 0)) >= 0.65 and not i.get("contradiction")]
            needs = len(strong) < min_verified_evidence and cycle_count < max_cycles
            return {
                "needs_more_research": needs,
                "focus_query": "",
                "ask_user": False,
                "question": "",
                "rationale": "mock_decision",
            }

        compact = []
        for item in evidence_items[:8]:
            compact.append(
                {
                    "title": item.get("title", ""),
                    "score": round(float(item.get("verification_score", 0)), 3),
                    "contradiction": bool(item.get("contradiction", False)),
                    "summary": str(item.get("summary", ""))[:220],
                }
            )
        prompt = (
            "Return JSON only with keys: needs_more_research (boolean), focus_query (string), "
            "ask_user (boolean), question (string), rationale (string). "
            "If evidence is weak/ambiguous and user clarification would help, set ask_user=true.\n"
            "Policy: early cycles should be exploratory and hypothesis-generating; "
            "later cycles should tighten into verification and structured synthesis.\n"
            f"Question: {query}\nCycle: {cycle_count}/{max_cycles}\n"
            f"Min verified evidence needed: {min_verified_evidence}\n"
            f"Evidence snapshot: {json.dumps(compact)}"
        )
        text = await self._chat(prompt, temperature=0.0, route=route)
        obj = self._parse_json_object(text)
        return {
            "needs_more_research": bool(obj.get("needs_more_research", False)),
            "focus_query": str(obj.get("focus_query", "")).strip(),
            "ask_user": bool(obj.get("ask_user", False)),
            "question": str(obj.get("question", "")).strip(),
            "rationale": str(obj.get("rationale", "")).strip(),
        }

    async def label_clusters(self, clusters: dict[int, list[str]]) -> dict[int, str]:
        """Single-pass LLM call: batch all cluster titles → 2-word category names."""
        if self.use_mock:
            return {k: f"Topic {k + 1}" for k in clusters}

        lines = []
        for k, titles in sorted(clusters.items()):
            sample = "; ".join(t[:60] for t in titles[:5])
            lines.append(f"Cluster {k}: {sample}")

        prompt = (
            "You are labeling clusters of research source titles.\n"
            "Return ONLY a JSON object mapping each cluster number (as string key) to a concise 2-3 word category name.\n"
            "Be specific and descriptive, not generic. Examples: 'Central Bank Policy', 'AI Chip Design', 'Climate Models'.\n\n"
            "Clusters:\n" + "\n".join(lines) + "\n\n"
            "Return JSON like: {\"0\": \"Label One\", \"1\": \"Label Two\", ...}"
        )
        # Use cheapest available model — this is a lightweight classification call
        cheap_route: dict[str, Any] | None = None
        if self.groq_api_key not in {"", "stub"}:
            cheap_route = {"provider": "groq", "model": self.model_groq_backup or self.model_groq, "provider_locked": False}
        elif self.anthropic_api_key not in {"", "stub"}:
            cheap_route = {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "provider_locked": False}

        try:
            text = await self._chat(prompt, temperature=0.0, route=cheap_route)
            obj = self._parse_json_object(text)
            if obj:
                return {int(k): str(v).strip() for k, v in obj.items() if str(k).isdigit()}
        except Exception:
            pass
        return {k: f"Topic {k + 1}" for k in clusters}

    async def _chat(
        self,
        prompt: str,
        temperature: float = 0.2,
        route: dict[str, Any] | None = None,
        max_completion_tokens: int = 1200,
    ) -> str:
        errors = []
        if route and route.get("provider"):
            provider = route.get("provider")
            model = route.get("model")
            provider_locked = bool(route.get("provider_locked", False))
            try:
                if provider == "together":
                    return await self._chat_deepseek(prompt, temperature, model_override=model, max_completion_tokens=max_completion_tokens)
                if provider == "groq":
                    return await self._chat_groq(prompt, temperature, model_override=model, max_completion_tokens=max_completion_tokens)
                if provider == "gemini":
                    return await self._chat_gemini(prompt, temperature, model_override=model, max_completion_tokens=max_completion_tokens)
                if provider == "openai":
                    return await self._chat_openai(prompt, temperature, model_override=model, max_completion_tokens=max_completion_tokens)
                if provider in {"anthropic", "claude"}:
                    return await self._chat_anthropic(prompt, temperature, model_override=model, max_completion_tokens=max_completion_tokens)
            except Exception as exc:
                msg = str(exc).strip() or repr(exc)
                errors.append(f"preferred route failed: {type(exc).__name__}: {msg}")
                if provider_locked:
                    raise RuntimeError("; ".join(errors))
        # Fallback chain: try providers in order of reliability
        fallback_chain = [
            ("groq", self.groq_api_key, self._chat_groq),
            ("anthropic", self.anthropic_api_key, self._chat_anthropic),
            ("openai", self.openai_api_key, self._chat_openai),
            ("together", self.deepseek_api_key, self._chat_deepseek),
            ("gemini", self.gemini_api_key, self._chat_gemini),
        ]
        for name, key, fn in fallback_chain:
            if key in {"", "stub"}:
                continue
            try:
                return await fn(prompt, temperature, max_completion_tokens=max_completion_tokens)
            except Exception as exc:
                msg = str(exc).strip() or repr(exc)
                errors.append(f"{name} failed: {type(exc).__name__}: {msg}")
        raise RuntimeError("All live providers failed: " + "; ".join(errors))

    async def _chat_openai(
        self,
        prompt: str,
        temperature: float,
        model_override: str | None = None,
        max_completion_tokens: int = 1200,
    ) -> str:
        payload = {
            "model": model_override or self.model_openai,
            "messages": [
                {"role": "system", "content": "You are a precise research agent. Follow requested output format exactly."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_completion_tokens": max(256, int(max_completion_tokens)),
        }
        headers = {"Authorization": f"Bearer {self.openai_api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=60) as client:
            res = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            res.raise_for_status()
            data = res.json()
            msg = data.get("choices", [{}])[0].get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                return content
            raise RuntimeError(f"OpenAI response missing text content: {str(data)[:320]}")

    async def _chat_anthropic(
        self,
        prompt: str,
        temperature: float,
        model_override: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> str:
        payload = {
            "model": model_override or self.model_anthropic,
            "max_tokens": max(256, min(8192, int(max_completion_tokens))),
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            "system": "You are a precise research agent. Think step by step when needed. Follow requested output format exactly.",
        }
        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        await _anthropic_limiter.acquire()
        try:
            client = httpx.AsyncClient(timeout=90)
            try:
                for attempt in range(3):
                    res = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
                    if res.status_code == 429:
                        retry_after = float(res.headers.get("retry-after", 2 * (attempt + 1)))
                        await asyncio.sleep(min(retry_after, 10))
                        continue
                    res.raise_for_status()
                    data = res.json()
                    out = []
                    for blk in data.get("content", []):
                        if isinstance(blk, dict) and blk.get("type") == "text":
                            out.append(str(blk.get("text", "")))
                    txt = "\n".join(out).strip()
                    if txt:
                        return txt
                    raise RuntimeError(f"Anthropic response missing text content: {str(data)[:320]}")
                raise RuntimeError("Anthropic rate limited after 3 retries")
            finally:
                await client.aclose()
        finally:
            _anthropic_limiter.release()

    async def _chat_deepseek(
        self,
        prompt: str,
        temperature: float,
        model_override: str | None = None,
        max_completion_tokens: int = 1200,
    ) -> str:
        payload = {
            "model": model_override or self.model_primary,
            "messages": [
                {"role": "system", "content": "You are a precise research agent. Follow requested output format exactly."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max(256, int(max_completion_tokens)),
        }
        headers = {"Authorization": f"Bearer {self.deepseek_api_key}", "Content-Type": "application/json"}
        last_exc = "unknown error"
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    res = await client.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload)
                    res.raise_for_status()
                    data = res.json()
                    choice = data.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    content = message.get("content")
                    if isinstance(content, str):
                        if content.strip():
                            return content
                    reasoning = message.get("reasoning")
                    if isinstance(reasoning, str) and reasoning.strip():
                        return reasoning
                    if isinstance(content, list):
                        txt = []
                        for chunk in content:
                            if isinstance(chunk, dict) and "text" in chunk:
                                txt.append(str(chunk["text"]))
                        merged = "\n".join(txt).strip()
                        if merged:
                            return merged
                    alt = choice.get("text")
                    if isinstance(alt, str) and alt.strip():
                        return alt
                    raise RuntimeError(f"Together response missing text content: {data}")
            except Exception as exc:
                last_exc = str(exc)
                if attempt < 2:
                    await asyncio.sleep(0.6 * (attempt + 1))
                    continue
                raise RuntimeError(last_exc)

    async def _chat_gemini(
        self,
        prompt: str,
        temperature: float,
        model_override: str | None = None,
        max_completion_tokens: int = 1200,
    ) -> str:
        model = model_override or self.model_fallback
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        params = {"key": self.gemini_api_key}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max(256, int(max_completion_tokens))},
        }
        async with httpx.AsyncClient(timeout=60) as client:
            res = await client.post(url, params=params, json=payload)
            res.raise_for_status()
            data = res.json()
            return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

    async def _chat_groq(
        self,
        prompt: str,
        temperature: float,
        model_override: str | None = None,
        max_completion_tokens: int = 1200,
    ) -> str:
        models: list[str] = []
        for mdl in [
            model_override or self.model_groq,
            self.model_groq_backup,
            self.model_groq_qwen32b,
            self.model_groq_llama_scout,
        ]:
            if mdl and mdl not in models:
                models.append(mdl)
        if not models:
            models = [self.model_groq]
        headers = {"Authorization": f"Bearer {self.groq_api_key}", "Content-Type": "application/json"}
        last_error = "unknown groq error"
        async with httpx.AsyncClient(timeout=30) as client:
            for mdl in models:
                prompts = [
                    prompt,
                    prompt + "\n\nIMPORTANT: Return plain text only. Do NOT call tools/functions.",
                ]
                for p in prompts:
                    payload = {
                        "model": mdl,
                        "messages": [
                            {"role": "system", "content": "You are a precise research agent. Follow requested output format exactly. Never call tools or functions."},
                            {"role": "user", "content": p},
                        ],
                        "temperature": temperature,
                        "max_completion_tokens": max(256, int(max_completion_tokens)),
                    }
                    try:
                        res = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
                        res.raise_for_status()
                        data = res.json()
                        choice = data.get("choices", [{}])[0]
                        msg = choice.get("message", {})
                        content = msg.get("content")
                        if isinstance(content, str) and content.strip():
                            return content
                        reasoning = msg.get("reasoning")
                        if isinstance(reasoning, str) and reasoning.strip():
                            return reasoning
                        alt = choice.get("text")
                        if isinstance(alt, str) and alt.strip():
                            return alt
                        raise RuntimeError(f"Groq response missing text content for model={mdl}: {str(data)[:320]}")
                    except httpx.HTTPStatusError as exc:
                        body = ""
                        try:
                            body = exc.response.text[:500]
                        except Exception:
                            body = ""
                        last_error = f"groq {exc.response.status_code} model={mdl}: {body or str(exc)}"
                        if "tool_use_failed" in body or "Tool choice is none, but model called a tool" in body:
                            continue
                        break
                    except Exception as exc:
                        last_error = str(exc).strip() or repr(exc)
                        break
        raise RuntimeError(last_error)

    @staticmethod
    def _normalize_agent_config(obj: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize agent_config returned by the planner LLM."""
        if not isinstance(obj, dict):
            return {}

        def _valid_route(r: Any) -> dict[str, Any] | None:
            if isinstance(r, dict) and r.get("provider") and r.get("model"):
                return {"provider": str(r["provider"]), "model": str(r["model"]), "provider_locked": False}
            return None

        raw_routes = obj.get("model_routes", {}) or {}
        model_routes: dict[str, Any] = {}
        for role in ("planner", "researcher", "verifier", "analyst", "writer"):
            r = _valid_route(raw_routes.get(role))
            if r:
                model_routes[role] = r

        researcher_routes: list[dict[str, Any]] = []
        for r in (obj.get("researcher_routes") or []):
            vr = _valid_route(r)
            if vr:
                researcher_routes.append(vr)

        try:
            researchers = min(6, max(1, int(obj.get("researchers", len(researcher_routes) or 2))))
        except (TypeError, ValueError):
            researchers = 2
        try:
            verifier_passes = min(3, max(1, int(obj.get("verifier_passes", 1))))
        except (TypeError, ValueError):
            verifier_passes = 1
        try:
            min_verified = min(6, max(1, int(obj.get("min_verified_evidence", 2))))
        except (TypeError, ValueError):
            min_verified = 2
        try:
            raw_score = obj.get("analyst_min_score")
            analyst_min_score = max(0.0, min(1.0, float(raw_score))) if raw_score is not None else None
        except (TypeError, ValueError):
            analyst_min_score = None

        ask_hitl = LLMAdapter._as_bool(obj.get("ask_hitl", False))
        hitl_question = str(obj.get("hitl_question", "")).strip()[:500]
        writer_sections = LLMAdapter._as_str_list(obj.get("writer_sections", []))

        try:
            search_results_per_query = min(15, max(5, int(obj.get("search_results_per_query", 10))))
        except (TypeError, ValueError):
            search_results_per_query = 10
        try:
            url_fetch_limit = min(30, max(10, int(obj.get("url_fetch_limit", 20))))
        except (TypeError, ValueError):
            url_fetch_limit = 20
        try:
            docs_per_subtask = min(12, max(3, int(obj.get("docs_per_subtask", 8))))
        except (TypeError, ValueError):
            docs_per_subtask = 8

        return {
            "researchers": researchers,
            "verifier_passes": verifier_passes,
            "min_verified_evidence": min_verified,
            "analyst_min_score": analyst_min_score,
            "ask_hitl": ask_hitl,
            "hitl_question": hitl_question,
            "writer_sections": writer_sections,
            "model_routes": model_routes,
            "researcher_routes": researcher_routes[:researchers],
            "search_results_per_query": search_results_per_query,
            "url_fetch_limit": url_fetch_limit,
            "docs_per_subtask": docs_per_subtask,
            "rationale": str(obj.get("rationale", "")).strip()[:300],
        }

    @staticmethod
    def _mock_plan(query: str) -> list[dict]:
        return [
            {
                "id": "p1",
                "description": "Scope and decompose the research question",
                "subquestions": [f"What is the exact scope of: {query}?"],
                "tools": ["web_search", "open_url"],
                "expected_cost_tokens": 900,
                "priority": 1,
                "status": "pending",
            },
            {
                "id": "p2",
                "description": "Collect supporting evidence and contradictory viewpoints",
                "subquestions": ["What evidence supports or challenges the core claims?"],
                "tools": ["web_search", "open_url", "summarize_text", "qdrant_upsert"],
                "expected_cost_tokens": 1800,
                "priority": 2,
                "status": "pending",
            },
            {
                "id": "p3",
                "description": "Synthesize evidence into final output with citations",
                "subquestions": ["How should findings be ranked by confidence and impact?"],
                "tools": ["qdrant_search"],
                "expected_cost_tokens": 1200,
                "priority": 3,
                "status": "pending",
            },
        ]

    @staticmethod
    def _parse_json_array(text: str) -> list[dict[str, Any]]:
        if not text:
            return []
        cleaned = LLMAdapter._extract_json_block(text)
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            return []
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            maybe = obj.get("plan")
            if isinstance(maybe, list):
                return maybe
        return []

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        if not text:
            return {}
        cleaned = LLMAdapter._extract_json_block(text)
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            return {}
        return obj if isinstance(obj, dict) else {}

    @staticmethod
    def _parse_json_object_loose(text: str) -> dict[str, Any]:
        """Best-effort salvage when model returns near-JSON plan payloads."""
        if not text:
            return {}
        m = re.search(r'(\{[\s\S]*"plan"\s*:\s*\[[\s\S]*\])', text)
        if not m:
            return {}
        candidate = m.group(1).strip()
        # Try closing object if model omitted trailing braces.
        if not candidate.endswith("}"):
            candidate = candidate + "}"
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return {}
        return {}

    @staticmethod
    def _extract_json_block(text: str) -> str:
        text = text.strip()
        # Extract content from markdown code blocks (```json ... ``` or ``` ... ```)
        m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        decoder = json.JSONDecoder()
        for start in range(len(text)):
            if text[start] not in "{[":
                continue
            try:
                _, end = decoder.raw_decode(text[start:])
                return text[start : start + end]
            except json.JSONDecodeError:
                continue
        return text

    @staticmethod
    def _coerce_plan_from_text(text: str, query: str) -> list[dict[str, Any]]:
        lines = [l.strip(" -*\t") for l in text.splitlines() if l.strip()]
        candidates = [l for l in lines if any(k in l.lower() for k in ("verify", "check", "find", "compare", "analyze", "synthesize"))]
        if not candidates:
            return []
        out = []
        for i, line in enumerate(candidates[:4], start=1):
            desc = re.sub(r"^[0-9]+[.)]\s*", "", line)
            out.append(
                {
                    "id": f"p{i}",
                    "description": desc[:180],
                    "subquestions": [f"How does this step answer: {query}?"],
                    "tools": ["web_search", "open_url", "playwright_fetch"],
                    "expected_cost_tokens": 1200,
                    "priority": i,
                    "status": "pending",
                }
            )
        return out

    @staticmethod
    def _coerce_summary_from_text(text: str) -> tuple[str, list[str]] | None:
        cleaned = (text or "").strip()
        if not cleaned:
            return None
        cleaned = re.sub(r"<think>.*?</think>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return None
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
        summary = " ".join(sentences[:3]).strip()
        claims: list[str] = []
        for s in sentences:
            low = s.lower()
            if any(k in low for k in ["claim", "evidence", "according to", "shows", "reported", "states", "found"]):
                claims.append(s)
            if len(claims) >= 3:
                break
        if not claims and sentences:
            claims = sentences[: min(3, len(sentences))]
        if not summary:
            return None
        return summary[:1200], [c[:300] for c in claims[:3]]

    @staticmethod
    def _extract_urls(text: str) -> list[str]:
        if not text:
            return []
        urls = re.findall(r"https?://[^\s)\]}\"'>,]+", text)
        out: list[str] = []
        for u in urls:
            if u not in out:
                out.append(u)
        return out

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            t = value.strip().lower()
            if t in {"true", "yes", "y", "1"}:
                return True
            if t in {"false", "no", "n", "0"}:
                return False
        return False

    @staticmethod
    def _as_str_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str) and value.strip():
            parts = [p.strip(" -*\t") for p in re.split(r"[\n;,]", value) if p.strip()]
            return parts
        return []

    @staticmethod
    def _normalize_verifier_object(obj: dict[str, Any], claims: list[str], raw_text: str = "") -> dict[str, Any] | None:
        if not isinstance(obj, dict) or not obj:
            return None
        score_raw = obj.get("verification_score", obj.get("score", obj.get("confidence")))
        score: float | None = None
        try:
            if score_raw is not None and score_raw != "":
                score = float(score_raw)
        except Exception:
            score = None
        contradiction = LLMAdapter._as_bool(obj.get("contradiction", obj.get("is_contradiction", obj.get("conflict"))))
        rationale = str(obj.get("rationale", obj.get("reason", obj.get("explanation", "")))).strip()
        supported_claims = LLMAdapter._as_str_list(obj.get("supported_claims", obj.get("supported", [])))
        missing_evidence = LLMAdapter._as_str_list(obj.get("missing_evidence", obj.get("gaps", [])))
        citations = LLMAdapter._as_str_list(obj.get("citations", obj.get("sources", obj.get("urls", []))))
        if raw_text:
            for u in LLMAdapter._extract_urls(raw_text):
                if u not in citations:
                    citations.append(u)

        if score is None:
            score = 0.45
        score = max(0.0, min(1.0, score))
        if not rationale:
            rationale = "normalized verifier output"
        if not supported_claims and claims:
            supported_claims = [str(c).strip() for c in claims[:1] if str(c).strip()]
        return {
            "verification_score": score,
            "contradiction": contradiction,
            "rationale": rationale[:500],
            "supported_claims": supported_claims[:5],
            "missing_evidence": missing_evidence[:5],
            "citations": citations[:8],
        }

    @staticmethod
    def _coerce_verification_from_text(text: str, claims: list[str]) -> dict[str, Any] | None:
        cleaned = (text or "").strip()
        if not cleaned:
            return None
        cleaned = re.sub(r"<think>.*?</think>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
        low = cleaned.lower()
        score = 0.45
        m = re.search(r"(?:score|confidence|verification_score)\s*[:=]\s*([01](?:\.\d+)?)", low)
        if m:
            try:
                score = float(m.group(1))
            except Exception:
                score = 0.45
        contradiction = any(k in low for k in ["contradict", "conflict", "inconsistent"])
        m_bool = re.search(r"(?:contradiction|conflict)\s*[:=]\s*(true|false|yes|no|1|0)", low)
        if m_bool:
            contradiction = m_bool.group(1) in {"true", "yes", "1"}
        citations = LLMAdapter._extract_urls(cleaned)
        rationale = "coerced verifier text output"
        if len(cleaned) < 320:
            rationale = cleaned
        return {
            "verification_score": max(0.0, min(1.0, score)),
            "contradiction": contradiction,
            "rationale": rationale[:500],
            "supported_claims": [str(c).strip() for c in claims[:1] if str(c).strip()],
            "missing_evidence": [],
            "citations": citations[:8],
        }
