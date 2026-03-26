from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from services.swarm_engine.settings import settings


_NOISE_PATHS = {"/healthz", "/api/v2/heartbeat", "/api/v1/heartbeat", "/_ping"}


class _NoHealthz(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in _NOISE_PATHS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Add filter after uvicorn fully configures its loggers
    logging.getLogger("uvicorn.access").addFilter(_NoHealthz())
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(router)


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}
