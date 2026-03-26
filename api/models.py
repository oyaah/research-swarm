from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str = Field(min_length=3)
    mode: Literal["single", "portfolio"] = "single"
    budget_mode: Literal["low", "balanced", "high"] = "balanced"
    depth: Literal["quick", "standard", "deep"] = "standard"
    thinking_depth: Literal["shallow", "standard", "deep"] = "standard"
    report_length: Literal["brief", "standard", "comprehensive"] = "standard"
    max_steps: int | None = None
    show_thinking: bool = True
    provider_pref: Literal["auto", "mixed", "groq", "together", "gemini", "openai", "anthropic"] = "auto"
    lane_preference: Literal["auto", "both", "fast", "deep"] = "auto"
    detail_level: Literal["compact", "brief", "detail"] = "brief"
    preference_text: str = ""
    cross_session_memory: bool = False


class ResearchResponse(BaseModel):
    session_id: str
    status: str


class HitlResponseRequest(BaseModel):
    answer: str


class SteerRequest(BaseModel):
    message: str = Field(min_length=2)
