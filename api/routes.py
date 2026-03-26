from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.mcp_client import get_tool_schema
from api.models import HitlResponseRequest, ResearchRequest, ResearchResponse, SteerRequest
from services.logging import get_logger
from services.swarm_engine.portfolio_engine import PortfolioPreferences
from services.swarm_engine.runtime import new_session_id, resume_session, run_portfolio_session, run_session, runtime
from services.supabase_store import supabase_store

router = APIRouter()
logger = get_logger("api.routes")


@router.post("/v1/research", response_model=ResearchResponse)
async def start_research(request: ResearchRequest) -> ResearchResponse:
    session_id = new_session_id()
    logger.info("start_research", extra={"event": "start_research", "session_id": session_id})
    if request.mode == "portfolio":
        prefs = PortfolioPreferences(
            budget_mode=request.budget_mode,
            depth=request.depth,
            provider_pref=request.provider_pref,
            lane_preference=request.lane_preference,
            detail_level=request.detail_level,
            preference_text=request.preference_text,
        )
        asyncio.create_task(
            run_portfolio_session(
                session_id=session_id,
                user_query=request.query,
                prefs=prefs,
                cross_session_memory=request.cross_session_memory,
            )
        )
    else:
        asyncio.create_task(
            run_session(
                session_id=session_id,
                user_query=request.query,
                cross_session_memory=request.cross_session_memory,
                budget_mode=request.budget_mode,
                depth=request.depth,
                provider_pref=request.provider_pref,
                preference_text=request.preference_text,
            )
        )
    return ResearchResponse(session_id=session_id, status="started")


@router.post("/v1/research/{session_id}/resume", response_model=ResearchResponse)
async def resume_research(session_id: str, body: HitlResponseRequest) -> ResearchResponse:
    if session_id not in runtime.status:
        if supabase_store.session_exists(session_id):
            runtime.status[session_id] = "running"
            runtime.final_state[session_id] = runtime.final_state.get(session_id, {"session_id": session_id})
            _ = runtime.queues[session_id]
            asyncio.create_task(resume_session(session_id=session_id, answer=body.answer))
            logger.info("resume_persisted", extra={"event": "resume_persisted", "session_id": session_id})
            return ResearchResponse(session_id=session_id, status="resumed")
        raise HTTPException(status_code=404, detail="Unknown session_id")
    if runtime.status.get(session_id) != "waiting_hitl":
        return ResearchResponse(session_id=session_id, status=runtime.status.get(session_id, "running"))
    asyncio.create_task(resume_session(session_id=session_id, answer=body.answer))
    logger.info("resume_active", extra={"event": "resume_active", "session_id": session_id})
    return ResearchResponse(session_id=session_id, status="resumed")


@router.get("/v1/research/{session_id}")
async def get_status(session_id: str) -> dict:
    if session_id not in runtime.status:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return {
        "session_id": session_id,
        "status": runtime.status[session_id],
        "final_state": runtime.final_state.get(session_id, {}),
    }


@router.get("/v1/sessions")
async def list_sessions() -> dict:
    merged: dict[str, dict] = {}
    for sid in sorted(runtime.status.keys()):
        merged[sid] = {
            "session_id": sid,
            "status": runtime.status.get(sid, "unknown"),
            "source": "runtime",
            "active": True,
        }
    for row in supabase_store.list_recent_sessions(limit=200):
        sid = row.get("session_id", "")
        if not sid:
            continue
        if sid not in merged:
            merged[sid] = {
                "session_id": sid,
                "status": row.get("status", "unknown"),
                "source": "supabase",
                "active": False,
                "updated_at": row.get("updated_at", ""),
            }
    items = list(merged.values())
    items.sort(key=lambda x: (0 if x.get("active") else 1, str(x.get("session_id"))))
    return {"sessions": items}


@router.post("/v1/research/{session_id}/steer", response_model=ResearchResponse)
async def steer_research(session_id: str, body: SteerRequest) -> ResearchResponse:
    if session_id not in runtime.status:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    runtime.steer_messages[session_id].append(body.message.strip())
    logger.info("steer", extra={"event": "steer", "session_id": session_id})
    await runtime.emit(
        session_id,
        {
            "type": "trace",
            "node_name": "HITL_Steer",
            "payload": {"message": body.message.strip()},
        },
    )
    return ResearchResponse(session_id=session_id, status=runtime.status.get(session_id, "running"))


@router.get("/v1/stream/{session_id}")
async def stream_session(session_id: str) -> EventSourceResponse:
    if session_id not in runtime.queues:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    async def event_generator():
        while True:
            event = await runtime.queues[session_id].get()
            yield {"event": event.get("type", "trace"), "data": json.dumps(event)}
            if event.get("type") == "done":
                break

    return EventSourceResponse(event_generator())


@router.get("/v1/tools")
async def list_tools() -> dict:
    return {"tools": get_tool_schema()}
